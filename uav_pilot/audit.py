"""Audit saved data and replay selected policies without changing original experiment outputs."""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .core import Model, Objective, POLICIES, SEARCH_POLICIES, Problem, expected_metrics


def load_problem(row):
    values = dict(row)
    values["edges"] = tuple(tuple(edge) for edge in values["edges"])
    values["viewpoints"] = tuple(tuple(group) for group in values["viewpoints"])
    values["prior"] = tuple(Fraction(p) for p in values["prior"])
    values["observation_rows"] = tuple((n, Fraction(a), Fraction(b)) for n, a, b in values["observation_rows"])
    return Problem(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    problems = {row["task_id"]: load_problem(row) for row in (
        json.loads(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines())}
    assert len(problems) == manifest["task_count"]
    mismatched = [name for name, digest in manifest["artifact_sha256"].items()
                  if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    assert not mismatched, f"Artifact hashes differ: {mismatched}"
    model_files = ["uav_pilot/core.py", "uav_pilot/scenarios.py", "uav_pilot/simulation.py", "uav_pilot/run.py"]
    source_root = Path(__file__).resolve().parents[1]
    assert all(hashlib.sha256((source_root / name).read_bytes()).hexdigest()
               == manifest["source_sha256"][str(Path(name))] for name in model_files), "Experiment code differs from manifest"
    with (root / "expected.csv").open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    with (root / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        episodes = list(csv.DictReader(handle))
    assert len(expected) == manifest["policy_evaluations"]
    assert len(episodes) == manifest["rollout_count"]
    assert len({(r["task_id"], r["question_penalty"], r["method"]) for r in expected}) == len(expected)
    for r in expected:
        if r["status"] == "complete":
            assert 0 <= float(r["success"]) <= 1 + 1e-12
    for r in episodes:
        assert 0 <= int(r["energy"]) <= problems[r["task_id"]].budget
        assert int(r["returned"]) == 1
    public_actions = {}
    trace_count = 0
    with (root / "traces.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            trace_count += 1
            for entry in episode["trace"]:
                key = (episode["task_id"], episode["question_penalty"], episode["method"],
                       entry["node"], entry["budget"], entry["steps_left"], tuple(entry["belief"]),
                       tuple(entry["seen"]), entry["asked"])
                assert key not in public_actions or public_actions[key] == entry["action"], "Nonanticipativity violation"
                public_actions[key] = entry["action"]
    assert trace_count == len(episodes)
    # Replay eight tasks at a fixed, declared operating point, including all methods.
    task_ids = set(list(problems)[:8])
    penalty = 20 if 20 in cfg["question_penalties"] else cfg["question_penalties"][0]
    replayed = 0
    max_difference = 0.0
    for row in expected:
        if row["task_id"] not in task_ids or float(row["question_penalty"]) != penalty or row["status"] != "complete":
            continue
        model = Model(problems[row["task_id"]])
        objective = Objective(cfg["failure_cost"], cfg["energy_weight"], penalty)
        policy = POLICIES[row["method"]](model, objective)
        result = expected_metrics(policy)
        for key in ("success", "energy", "questions", "observations", "aborts"):
            difference = abs(getattr(result, key) - float(row[key]))
            max_difference = max(max_difference, difference)
            assert difference <= 1e-12
        assert abs(objective.value(result) - float(row["objective"])) <= 1e-12
        replayed += 1
        if isinstance(policy, SEARCH_POLICIES):
            policy.close()
    from PIL import Image
    figures = {}
    for path in root.glob("*.png"):
        with Image.open(path) as img:
            figures[path.name] = {"pixels": img.size, "dpi": img.info.get("dpi"), "mode": img.mode}
    report = {"status": "passed", "run_directory": str(root),
              "artifact_hashes_checked": len(manifest["artifact_sha256"]),
              "core_source_hashes_match": True, "policy_rows": len(expected), "episode_rows": len(episodes),
              "trace_episodes": trace_count, "distinct_public_states_checked": len(public_actions),
              "nonanticipativity": "passed for logged public states",
              "replayed_policy_cases": replayed, "max_replay_metric_difference": max_difference,
              "resource_and_return_checks": "passed within the discrete model", "figures": figures}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
