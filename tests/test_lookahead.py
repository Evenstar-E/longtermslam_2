import csv
import json
from fractions import Fraction as F
from pathlib import Path
import unittest

from uav_pilot.audit import load_problem
from uav_pilot.core import (
    Model, Objective, MyopicVOIPolicy, ExactPolicy, DepthLimitedPolicy,
    Lookahead2Policy, Lookahead3Policy, POLICIES, expected_metrics,
)
from uav_pilot.scenarios import generate_tasks, worked_example
from uav_pilot.simulation import rollout


class LookaheadTests(unittest.TestCase):
    def development_tasks(self):
        return generate_tasks({"seed": 180, "cohort": "test", "development_layouts": 2,
                               "horizon": 6}, "development")

    def test_depth_one_reproduces_existing_voi(self):
        for problem in self.development_tasks():
            model = Model(problem)
            for penalty in (0, 20, 60):
                objective = Objective(question_penalty=penalty)
                one = DepthLimitedPolicy(model, objective, depth=1)
                old = MyopicVOIPolicy(model, objective)
                a, b = expected_metrics(one), expected_metrics(old)
                for key in ("success", "energy", "questions", "observations", "aborts"):
                    self.assertAlmostEqual(getattr(a, key), getattr(b, key), places=10)
                one.close()

    def test_full_horizon_matches_exact_objective(self):
        for problem in self.development_tasks():
            model, objective = Model(problem), Objective(question_penalty=20)
            full = DepthLimitedPolicy(model, objective, depth=problem.horizon)
            exact = ExactPolicy(model, objective)
            self.assertAlmostEqual(objective.value(expected_metrics(full)),
                                   objective.value(expected_metrics(exact)), places=9)
            full.close()
            exact.close()

    def test_two_steps_can_plan_move_then_observe(self):
        model = Model(worked_example(start="S", budget=140))
        objective = Objective(question_penalty=20)
        one, two = MyopicVOIPolicy(model, objective), Lookahead2Policy(model, objective)
        self.assertEqual(one.choose(model.initial()).kind, "ask")
        self.assertEqual(two.choose(model.initial()).label(), "move:P")
        self.assertLess(objective.value(expected_metrics(two)), objective.value(expected_metrics(one)))
        two.close()

    def test_exact_dominates_receding_lookahead(self):
        for problem in self.development_tasks():
            model, objective = Model(problem), Objective(question_penalty=60)
            exact = ExactPolicy(model, objective)
            optimal = objective.value(expected_metrics(exact))
            for cls in (Lookahead2Policy, Lookahead3Policy):
                candidate = cls(model, objective)
                self.assertLessEqual(optimal, objective.value(expected_metrics(candidate)) + 1e-8)
                candidate.close()
            exact.close()

    def test_no_information_does_not_trigger_wasted_observation(self):
        model = Model(worked_example(observation_rows=(("P", F(1, 2), F(1, 2)),)))
        for cls in (Lookahead2Policy, Lookahead3Policy):
            policy = cls(model, Objective())
            self.assertEqual(expected_metrics(policy).observations, 0)
            policy.close()

    def test_budget_and_no_initial_truth_dependence(self):
        for cls in (Lookahead2Policy, Lookahead3Policy):
            model = Model(worked_example())
            policy = cls(model, Objective(question_penalty=20))
            samples = [rollout(policy, seed) for seed in range(20)]
            self.assertEqual(len({trace[0]["action"] for _, trace in samples}), 1)
            self.assertEqual(len({row["private_true_z"] for row, _ in samples}), 2)
            for row, _ in samples:
                self.assertLessEqual(row["energy"], model.problem.budget)
                self.assertEqual(row["returned"], 1)
            policy.close()

    def test_timeout_and_invalid_depth(self):
        model = Model(worked_example())
        with self.assertRaises(ValueError):
            DepthLimitedPolicy(model, Objective(), depth=0)
        policy = Lookahead3Policy(model, Objective(), timeout_s=-1)
        with self.assertRaises(TimeoutError):
            policy.choose(model.initial())
        policy.close()

    @unittest.skipUnless(Path("results/pilot_v1/expected.csv").exists(), "Optional saved-v1 regression data absent")
    def test_saved_v1_baselines_unchanged(self):
        root = Path("results/pilot_v1")
        problems = [load_problem(json.loads(line)) for line in
                    (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[:6]]
        with (root / "expected.csv").open(encoding="utf-8", newline="") as handle:
            expected = {(r["task_id"], r["method"]): r for r in csv.DictReader(handle)
                        if float(r["question_penalty"]) == 20}
        for problem in problems:
            for method in ("map", "ask_first", "task_voi", "exact"):
                policy = POLICIES[method](Model(problem), Objective(question_penalty=20))
                result = expected_metrics(policy)
                row = expected[(problem.task_id, method)]
                for key in ("success", "energy", "questions", "observations", "aborts"):
                    self.assertAlmostEqual(getattr(result, key), float(row[key]), places=12)
                if hasattr(policy, "close"):
                    policy.close()


if __name__ == "__main__":
    unittest.main()
