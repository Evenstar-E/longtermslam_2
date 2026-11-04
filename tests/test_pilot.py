from dataclasses import replace
from fractions import Fraction as F
import unittest

from uav_pilot.core import (
    Action, Model, State, Objective, MAPPolicy, AskFirstPolicy, MyopicVOIPolicy,
    ExactPolicy, Policy, expected_metrics,
)
from uav_pilot.scenarios import worked_example, generate_tasks
from uav_pilot.simulation import rollout


class PilotTests(unittest.TestCase):
    def test_hand_computed_completion_costs(self):
        m = Model(worked_example())
        self.assertEqual([m.completion_cost("S", z) for z in (0, 1)], [85, 95])
        self.assertEqual([m.completion_cost("P", z) for z in (0, 1)], [65, 75])
        self.assertEqual([m.completion_cost("R", z) for z in (0, 1)], [53, 87])

    def test_bayes_update_matches_manual_example(self):
        m = Model(worked_example())
        branches = m.branches(m.initial(), Action("observe"))
        self.assertEqual(branches[0][0], F(59, 100))
        self.assertEqual(branches[0][2].belief[0], F(57, 59))
        self.assertEqual(branches[1][2].belief[1], F(38, 41))

    def test_last_query_boundary(self):
        self.assertEqual(F(90 - 75 - 6, 2), F(9, 2))
        for d, possible in ((4, True), (5, False)):
            p = worked_example(edges=(("S", "P", 20), ("P", "X", d),
                ("X", "R", 12-d), ("R", "A", 8), ("P", "B", 25)))
            m = Model(p)
            state = replace(m.initial(), node="X", budget=90-d)
            posterior_b = next(child for _, y, child in m.branches(state, Action("ask")) if y == 1)
            self.assertEqual(Action("finish", "B") in m.actions(posterior_b), possible)

    def test_uninformative_observation(self):
        m = Model(worked_example(observation_rows=(("P", F(1, 2), F(1, 2)),)))
        for _, _, state in m.branches(m.initial(), Action("observe")):
            self.assertEqual(state.belief, m.initial().belief)
        policy = ExactPolicy(m, Objective())
        self.assertAlmostEqual(expected_metrics(policy).observations, 0)

    def test_free_reliable_query(self):
        m = Model(worked_example(ask_cost=0, observation_rows=()))
        policy = ExactPolicy(m, Objective())
        metrics = expected_metrics(policy)
        self.assertEqual(metrics.success, 1)
        self.assertEqual(metrics.questions, 1)
        self.assertLessEqual(Objective().value(metrics), Objective().value(expected_metrics(MAPPolicy(m, Objective()))))

    def test_equivalent_interpretations_do_not_require_query(self):
        m = Model(worked_example(viewpoints=(("A",), ("A",))))
        for cls in (AskFirstPolicy, ExactPolicy):
            result = expected_metrics(cls(m, Objective()))
            self.assertEqual(result.success, 1)
            self.assertEqual(result.questions, 0)
            self.assertEqual(result.observations, 0)

    def test_initially_impossible_aborts(self):
        m = Model(worked_example(start="S", budget=30))
        for cls in (MAPPolicy, AskFirstPolicy, MyopicVOIPolicy, ExactPolicy):
            metrics = expected_metrics(cls(m, Objective()))
            self.assertEqual(metrics.success, 0)
        self.assertEqual(ExactPolicy(m, Objective()).choose(m.initial()).kind, "abort")
        self.assertEqual(expected_metrics(AskFirstPolicy(m, Objective())).questions, 0)

    def test_finite_horizon_and_return_reserve(self):
        m = Model(worked_example(horizon=1))
        after_ask = m.branches(m.initial(), Action("ask"))[0][2]
        self.assertEqual(m.actions(after_ask), (Action("abort"),))
        self.assertEqual(m.initial().steps, 1)
        for cls in (MAPPolicy, AskFirstPolicy, MyopicVOIPolicy, ExactPolicy):
            for seed in range(10):
                row, _ = rollout(cls(m, Objective()), seed)
                self.assertLessEqual(row["energy"], m.problem.budget)
                self.assertEqual(row["returned"], 1)

    def test_public_policy_input_does_not_contain_episode_truth(self):
        m = Model(worked_example())
        self.assertNotIn("true_z", m.problem.__dataclass_fields__)
        self.assertNotIn("true_z", State.__dataclass_fields__)
        p = ExactPolicy(m, Objective())
        paths = [rollout(p, seed) for seed in range(20)]
        self.assertEqual(len({row["private_true_z"] for row, _ in paths}), 2)
        # Same initial observable history must have identical first action for both truths.
        self.assertEqual(len({trace[0]["action"] for _, trace in paths}), 1)

    def test_exact_dominates_baselines_on_seeded_tasks(self):
        cfg = {"seed": 22, "development_layouts": 2, "horizon": 6}
        for problem in generate_tasks(cfg, "development"):
            m = Model(problem)
            for penalty in (0, 60):
                objective = Objective(question_penalty=penalty)
                exact = objective.value(expected_metrics(ExactPolicy(m, objective)))
                for cls in (MAPPolicy, AskFirstPolicy, MyopicVOIPolicy):
                    self.assertLessEqual(exact, objective.value(expected_metrics(cls(m, objective))) + 1e-8)

    def test_seeded_rollout_matches_enumerated_distribution(self):
        m = Model(worked_example())
        policy = MyopicVOIPolicy(m, Objective(question_penalty=20))
        exact = expected_metrics(policy)
        rows = [rollout(policy, seed)[0] for seed in range(1500)]
        observed = sum(r["success"] for r in rows) / len(rows)
        self.assertLess(abs(observed - exact.success), 0.04)
        first = rollout(policy, 123)[0]
        self.assertEqual(first, rollout(policy, 123)[0])

    def test_invalid_problem_is_rejected(self):
        with self.assertRaises(ValueError):
            Model(worked_example(prior=(F(1), F(1))))
        with self.assertRaises(ValueError):
            Model(worked_example(edges=(("S", "P", 0),)))
        with self.assertRaises(ValueError):
            Objective(failure_cost=-1)

    def test_conditional_policy_matches_hand_calculated_expectation(self):
        class ConditionalPolicy(Policy):
            def choose(self, state):
                if not state.seen:
                    return Action("observe")
                if max(state.belief) < F(95, 100) and not state.asked:
                    return Action("ask")
                return MAPPolicy(self.model, self.objective).choose(state)
        result = expected_metrics(ConditionalPolicy(Model(worked_example()), Objective()))
        self.assertAlmostEqual(result.energy, 74.26)
        self.assertAlmostEqual(result.success, 0.98)
        self.assertAlmostEqual(result.questions, 0.41)
        self.assertAlmostEqual(result.observations, 1.0)


if __name__ == "__main__":
    unittest.main()
