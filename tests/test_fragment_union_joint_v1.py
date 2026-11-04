import unittest
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.fragment_union_joint_v1 import DirectFactors, huber, key_matrix, union_of_masks


class FragmentUnionTests(unittest.TestCase):
    def model(self, masks, predicted, depth):
        model = DirectFactors.__new__(DirectFactors)
        model.cfg = dict(missing_support_penalty=12., occlusion_margin_m=.06,
                         ray_depth_sigma_m=.06, huber_transition=2., boundary_sigma_px=1.5)
        model.masks = [masks]; model.current = [dict(depth=depth)]
        model.predicted = lambda *args: predicted
        return model

    def test_occlusion_fragment_partition_invariance(self):
        expected = np.full((6, 8), 5., dtype=np.float32)
        depth = expected.copy(); depth[:, 3:5] = 3.
        left = np.zeros_like(depth, bool); left[:, :3] = True
        right = np.zeros_like(depth, bool); right[:, 5:] = True
        separate = self.model([left, right], expected, depth)
        joined = self.model([left|right], expected, depth)
        a = separate.projective_factor(0, 0, [0, 0], key_matrix(np.eye(4)), np.zeros((1, 3)))
        b = joined.projective_factor(0, 0, [0], key_matrix(np.eye(4)), np.zeros((1, 3)))
        self.assertEqual(a, b); self.assertEqual(a[0], 0.); self.assertEqual(a[1], 36)

    def test_real_motion_toward_camera_not_censored_as_foreground(self):
        expected = np.full((6, 8), 5., dtype=np.float32)
        mask = np.ones_like(expected, bool)
        model = self.model([mask], expected, expected-.3)
        loss, comparable = model.projective_factor(0, 0, [0], key_matrix(np.eye(4)), np.zeros((1, 3)))
        self.assertGreater(loss, 0.); self.assertEqual(comparable, 48)

    def test_model_outside_image_cannot_explain_assigned_pixels_for_free(self):
        expected = np.full((6, 8), np.inf, dtype=np.float32)
        mask = np.ones_like(expected, bool)
        model = self.model([mask], expected, np.full_like(expected, 5.))
        self.assertEqual(model.projective_factor(0, 0, [0], key_matrix(np.eye(4)), np.zeros((1, 3)))[0], 12.)

    def test_empty_observed_set_with_known_clear_depth_is_contradiction(self):
        expected = np.full((6, 8), 5., dtype=np.float32)
        model = self.model([np.zeros_like(expected, bool)], expected, expected+5.)
        self.assertEqual(model.projective_factor(0, 0, [-1], key_matrix(np.eye(4)), np.zeros((1, 3)))[0], 12.)

    def test_actual_point_model_allows_two_fragments_one_asset(self):
        model = DirectFactors.__new__(DirectFactors)
        points = np.array([[x, y, 5.] for x in [-.3, -.1, .1, .3] for y in [-.2, .2]])
        model.refs = [dict(points=points, group=0)]; model.n = 1
        model.trees = [cKDTree(points)]; model.points = [[points[:4], points[4:]]]
        model.frames = [[dict(group=0), dict(group=0)]]
        model.cfg = dict(sigma=.06, huber_transition=2., null_penalty=12., change_penalty=4.,
            motion_scale_m=.6, max_local_motion_m=.6, se3_translation_bound_m=3., se3_rotation_bound_deg=15.)
        model.evaluations = 0
        many = model.nearest_assignment(np.eye(4), np.zeros((1, 3)))
        one = model.nearest_assignment(np.eye(4), np.zeros((1, 3)), True)
        self.assertEqual(many, [[0, 0]])
        self.assertEqual(sum(j < 0 for j in one[0]), 1)
        self.assertEqual(model.energy('point_many', np.eye(4), np.zeros((1, 3)), many), 0.)
        self.assertEqual(model.energy('point_one_to_one', np.eye(4), np.zeros((1, 3)), one), 12.)

    def test_robust_loss_is_quadratic_near_zero_and_linear_in_tail(self):
        np.testing.assert_allclose(huber(np.array([0., 1., 2., 3., 4.])), [0., 1., 4., 8., 12.])


if __name__ == '__main__': unittest.main()
