import unittest
import numpy as np
from inspection_decisions.profiled_identity_actions_v1 import bounded_profile, expected_risk, report_distance, choose
from scripts.run_profiled_identity_actions_v1 import relative_motion


class ProfiledIdentityTests(unittest.TestCase):
    def test_shared_nuisance_removes_only_explainable_difference(self):
        r = np.array([2., -2., 3.])
        J = np.array([[2.], [-2.], [0.]])
        p = bounded_profile(r, J)
        self.assertAlmostEqual(p['raw'], 17.)
        self.assertAlmostEqual(p['profiled'], 9., places=5)
        self.assertLessEqual(abs(p['eta'][0]), 1.)

    def test_nuisance_cannot_be_independently_chosen_for_each_pixel(self):
        # The same pose parameter moves both measurements together. Marginal
        # independent correction could erase [1,-1], a shared correction cannot.
        p = bounded_profile(np.array([1., -1.]), np.ones((2, 1)))
        self.assertAlmostEqual(p['profiled'], 2.)

    def test_bounds_prevent_unlimited_pose_explanation(self):
        p = bounded_profile(np.array([4.]), np.array([[1.]]))
        self.assertAlmostEqual(p['profiled'], 9., places=5)
        self.assertGreaterEqual(p['eta'][0], -1.)

    def test_correlated_identity_report_and_action(self):
        def option(matches):
            return dict(best=dict(matches=[matches]), prediction=dict(assets=[
                dict(label=0, delta=[0., 0., 0.]) for _ in range(3)]))
        D = report_distance([option([(0, 0), (1, 1)]), option([(0, 1), (1, 2)])])
        self.assertEqual(D[0, 1], 1.)
        unresolved = expected_risk(np.zeros((2, 2)), D)
        resolved = expected_risk(np.array([[0., 20.], [20., 0.]]), D)
        self.assertAlmostEqual(unresolved['report_risk'], .5)
        self.assertLess(resolved['report_risk'], 1e-8)
        self.assertEqual(choose('profiled_report_risk', dict(actions=[
            dict(action=1, profiled_report_risk=.4), dict(action=2, profiled_report_risk=.1)])), 2)

    def test_relative_camera_action_is_map_gauge_equivariant(self):
        action = dict(body_translation_m=[1., 0., -2.], body_yaw_deg=14.)
        P = relative_motion(action); T = np.eye(4); T[:3, 3] = [3., 2., 1.]
        G = np.eye(4); G[:3, :3] = P[:3, :3]; G[:3, 3] = [1., 2., 3.]
        np.testing.assert_allclose(np.linalg.inv(G@T)@(G@T@P), P, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm((G@T@P)[:3, 3]-(G@T)[:3, 3]), np.sqrt(5.))


if __name__ == '__main__':
    unittest.main()
