"""Run a bounded, logged, synthetic experiment in a new output directory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

from .core import Model, Objective, POLICIES, SEARCH_POLICIES, expected_metrics
from .scenarios import generate_tasks
from .simulation import rollout


EXPECTED_FIELDS = [
    "task_id", "layout_id", "family", "split", "question_penalty", "method", "status",
    "success", "energy", "questions", "observations", "aborts", "objective",
    "root_plan_ms", "root_expanded_states", "expanded_states", "initial_action", "error",
]
EPISODE_FIELDS = [
    "task_id", "layout_id", "family", "split", "question_penalty", "method", "seed",
    "private_true_z", "success", "energy", "questions", "observations", "abort",
    "returned", "initial_true_feasible", "pre_capture_lost_recoverability", "steps",
]


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    p.add_argument("--split", choices=["development", "evaluation", "sensitivity"], default="evaluation")
    p.add_argument("--output", type=Path, required=True, help="New directory; never overwrites a run")
    p.add_argument("--max-tasks", type=int, help="Truncate only for smoke tests; recorded in manifest")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be positive")
    tasks = generate_tasks(cfg, args.split)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", cfg)
    source_root = Path(__file__).resolve().parents[1]
    sources = sorted((source_root / "uav_pilot").glob("*.py")) + sorted((source_root / "tests").glob("*.py"))
    sources += [args.config.resolve()]
    hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    deps = {name: version(name) for name in ("matplotlib", "numpy", "pillow")}
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "uav_pilot.run", *sys.argv[1:]],
        "python": sys.version, "platform": platform.platform(), "dependencies": deps,
        "source_sha256": hashes, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "split": args.split, "task_count": len(tasks), "max_tasks": args.max_tasks,
        "independent_layouts": len({p.layout_id for p in tasks}),
        "failure_cost": cfg["failure_cost"], "energy_weight": cfg["energy_weight"],
        "expected_metrics": "Enumeration of all outcomes in a finite discrete policy; not Monte Carlo estimates",
        "units": "Synthetic resource units, not calibrated Joules or battery capacity",
        "horizon": "Remaining macro decision count; finish includes travel/photo/return",
        "version_label": cfg.get("version_label", "pilot_v1"),
        "search_timeout_seconds": cfg["exact_timeout_seconds"],
        "search_budget_note": "Same per-policy solve timeout for exact and depth-limited search; not an equal-time truncation experiment",
        "information": "Two symbolic hypotheses; known map/prior/likelihood; reliable binary ask; one observation per site",
        "limitations": ["No VLM, frame parser, SLAM, aerodynamics or collision model",
                        "No novel solver claimed", "No human instructions or industrial prevalence evidence",
                        "Monte Carlo repeated seeds are not independent layouts"],
    }
    dump(output / "manifest.json", manifest)
    with (output / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    started = perf_counter()
    timeout_count = 0
    row_count = 0
    episode_count = 0
    expected_rows = []
    try:
        with ((output / "expected.csv").open("w", newline="", encoding="utf-8") as expected_file,
              (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
              (output / "traces.jsonl").open("w", encoding="utf-8") as trace_file):
            writer = csv.DictWriter(expected_file, fieldnames=EXPECTED_FIELDS)
            writer.writeheader()
            episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
            episode_writer.writeheader()
            for index, task in enumerate(tasks):
                if perf_counter() - started > cfg["run_timeout_seconds"]:
                    raise TimeoutError("Overall configured run deadline exceeded")
                model = Model(task)
                for penalty in cfg["question_penalties"]:
                    objective = Objective(cfg["failure_cost"], cfg["energy_weight"], penalty)
                    case_values = {}
                    for method in cfg["methods"]:
                        base = {"task_id": task.task_id, "layout_id": task.layout_id,
                                "family": task.family, "split": args.split,
                                "question_penalty": penalty, "method": method}
                        cls = POLICIES[method]
                        policy = (cls(model, objective, timeout_s=cfg["exact_timeout_seconds"])
                                  if issubclass(cls, SEARCH_POLICIES) else cls(model, objective))
                        try:
                            planning_start = perf_counter()
                            action = policy.choose(model.initial())
                            plan_ms = (perf_counter() - planning_start) * 1000
                            root_expansions = getattr(policy, "expansions", 0)
                            metrics = expected_metrics(policy)
                            value = objective.value(metrics)
                            case_values[method] = value
                            row = {**base, "status": "complete", **asdict(metrics), "objective": value,
                                   "root_plan_ms": plan_ms, "root_expanded_states": root_expansions,
                                   "expanded_states": getattr(policy, "expansions", 0),
                                   "initial_action": action.label(), "error": ""}
                            for repeat in range(cfg["rollout_repeats"]):
                                seed = cfg["seed"] + repeat
                                episode, trace = rollout(policy, seed)
                                episode_writer.writerow({**base, **episode})
                                episode_count += 1
                                # Full policy state/action/outcome trajectories retained for every rollout.
                                trace_file.write(json.dumps({**base, **episode, "trace": trace}, ensure_ascii=False) + "\n")
                        except TimeoutError as exc:
                            timeout_count += 1
                            row = {**base, "status": "timeout", "error": str(exc)}
                        finally:
                            if isinstance(policy, SEARCH_POLICIES):
                                policy.close()
                        writer.writerow(row)
                        expected_rows.append(row)
                        row_count += 1
                    if "exact" in case_values:
                        for method, value in case_values.items():
                            if case_values["exact"] > value + 1e-7:
                                raise AssertionError(f"Exact-policy dominance violation: {task.task_id}, {method}")
                expected_file.flush()
                episode_file.flush()
                trace_file.flush()
                dump(output / "progress.json", {"completed_tasks": index + 1, "total_tasks": len(tasks),
                     "elapsed_s": perf_counter() - started, "timeouts": timeout_count})
                if (index + 1) % 6 == 0 or index + 1 == len(tasks):
                    print(f"[{index+1}/{len(tasks)}] {row_count} policy evaluations, {episode_count} rollouts, "
                          f"{perf_counter()-started:.1f}s, timeouts={timeout_count}", flush=True)
        manifest.update(status="complete" if not timeout_count else "complete_with_timeouts",
                        elapsed_seconds=perf_counter() - started, timeouts=timeout_count,
                        policy_evaluations=row_count, rollout_count=episode_count,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        from .report import summarize, make_plots
        summarize(output, cfg, manifest, expected_rows)
        if not args.no_plots:
            make_plots(output, cfg, manifest, expected_rows)
        manifest["artifact_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file() and p.name not in {"manifest.json", "progress.json"}
        }
        dump(output / "manifest.json", manifest)
        print(f"Completed: {output}", flush=True)
    except Exception:
        manifest.update(status="failed", elapsed_seconds=perf_counter() - started,
                        error=traceback.format_exc())
        dump(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
