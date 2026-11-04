"""Reproduce v2 comparisons and selected explanatory policy trees from saved runs.

Example selection is post-hoc: largest depth-2 gap at penalty 20, then task ID.
This is an explanation of observed results, not another held-out evaluation.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .audit import load_problem
from .core import Model, Objective, POLICIES, expected_metrics


METRICS = ("success", "energy", "questions", "observations", "aborts", "objective")
AMBIGUOUS = {"fork", "observable", "pure_intent"}


def read_run(root):
    with (root / "expected.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(r["status"] == "complete" for r in rows), "Incomplete run"
    indexed = {(r["task_id"], float(r["question_penalty"]), r["method"]): r for r in rows}
    assert len(indexed) == len(rows), "Duplicate policy rows"
    tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows, indexed, tasks


def policy_tree(policy, state=None):
    state = state or policy.model.initial()
    action = policy.choose(state)
    result = {
        "node": state.node, "budget": state.budget, "steps": state.steps,
        "belief": [str(p) for p in state.belief], "action": action.label(),
        "immediate_resource_cost": policy.model.cost(state, action),
    }
    branches = policy.model.branches(state, action)
    if branches:
        result["branches"] = [
            {"probability": str(p), "observation": obs, "child": policy_tree(policy, child)}
            for p, obs, child in branches
        ]
    else:
        result["terminal_metrics"] = asdict(policy.model.terminal_metrics(state, action))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("results/pilot_v1"))
    parser.add_argument("--replay", type=Path, default=Path("results/pilot_v2_replay_v1"))
    parser.add_argument("--new", type=Path, default=Path("results/pilot_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output directory; existing evidence is never overwritten")
    old_rows, _, old_tasks = read_run(args.v1)
    replay_rows, replay_index, replay_tasks = read_run(args.replay)
    new_rows, new_index, new_tasks = read_run(args.new)
    assert old_tasks == replay_tasks, "Replay tasks differ from v1"
    regression_max = max(abs(float(row[m]) - float(replay_index[
        row["task_id"], float(row["question_penalty"]), row["method"]][m]))
        for row in old_rows for m in METRICS)
    assert regression_max <= 1e-10, "An old baseline changed"
    geometry = lambda tasks: {tuple(tuple(e) for e in t["edges"]) for t in tasks}
    overlap = len(geometry(old_tasks) & geometry(new_tasks))
    assert overlap == 0, "New and old graph edge costs overlap"
    summary = {
        "status": "passed", "tolerance": 1e-7,
        "regression": {"matched_policy_cases": len(old_rows),
                       "metrics": METRICS, "max_absolute_difference": regression_max,
                       "tasks_identical": True},
        "geometry": {"v1_unique": len(geometry(old_tasks)), "v2_unique": len(geometry(new_tasks)),
                     "overlap": overlap,
                     "note": "Different edge-cost layouts from the SAME graph topology and generator; not unseen topology generalization."},
        "runs": {},
    }
    examples = {}
    for name, rows, indexed, tasks, root in (
        ("new", new_rows, new_index, new_tasks, args.new),
        ("replay", replay_rows, replay_index, replay_tasks, args.replay),
    ):
        three = [r for r in rows if r["method"] == "lookahead_3"]
        diffs = {m: max(abs(float(r[m]) - float(indexed[
            r["task_id"], float(r["question_penalty"]), "exact"][m])) for r in three) for m in METRICS}
        assert max(diffs.values()) <= 1e-7, "Three-step equality claim does not hold"
        gaps = {}
        for method in ("task_voi", "lookahead_2", "lookahead_3"):
            pairs = [(r, float(r["objective"]) - float(indexed[
                r["task_id"], float(r["question_penalty"]), "exact"]["objective"]))
                for r in rows if r["method"] == method and r["family"] in AMBIGUOUS]
            gaps[method] = {
                "ambiguous_task_weight_pairs": len(pairs),
                "positive_gap_pairs": sum(d > 1e-7 for _, d in pairs),
                "max_objective_gap": max(d for _, d in pairs),
                "penalty20_gap_cases": [{"task_id": r["task_id"], "gap": d}
                    for r, d in pairs if float(r["question_penalty"]) == 20 and d > 1e-7],
            }
        summary["runs"][name] = {
            "three_vs_exact_cases": len(three), "max_metric_differences": diffs,
            "different_initial_actions": sum(r["initial_action"] != indexed[
                r["task_id"], float(r["question_penalty"]), "exact"]["initial_action"] for r in three),
            "gaps": gaps,
        }
        candidates = gaps["lookahead_2"]["penalty20_gap_cases"]
        if candidates:
            chosen = sorted(candidates, key=lambda x: (-x["gap"], x["task_id"]))[0]
            task = next(t for t in tasks if t["task_id"] == chosen["task_id"])
            example = {"selection": "Post-hoc largest depth-2 gap at penalty20; ties by task_id",
                       "problem": task, "policies": {}}
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            objective = Objective(config["failure_cost"], config["energy_weight"], 20)
            for method in ("task_voi", "lookahead_2", "lookahead_3", "exact"):
                policy = POLICIES[method](Model(load_problem(task)), objective)
                metrics = expected_metrics(policy)
                example["policies"][method] = {"expected_metrics": asdict(metrics),
                    "objective": objective.value(metrics), "tree": policy_tree(policy)}
                if hasattr(policy, "close"):
                    policy.close()
            examples[name] = example
    paths = [Path(__file__), Path(__file__).with_name("core.py")]
    paths += [root / f for root in (args.v1, args.replay, args.new)
              for f in ("expected.csv", "tasks.jsonl", "config.json")]
    summary["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True)
    for filename, payload in (("comparison.json", summary), ("example_policy_trees.json", examples)):
        (args.output / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
