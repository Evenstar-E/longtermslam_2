"""Negative model counterexamples, not evidence of task success or a fix."""
import unittest
import numpy as np
from inspection_decisions.fragment_union_joint_v1 import DirectFactors, key_matrix


def counterexample():
    model = DirectFactors.__new__(DirectFactors)
    left = np.zeros((8, 10), bool); left[:, :4] = True
    unrelated = np.zeros_like(left); unrelated[:, 6:] = True
    predicted = np.full(left.shape, np.inf, dtype=np.float32); predicted[left] = 5.
    depth = np.full(left.shape, 12., dtype=np.float32); depth[left] = 5.; depth[unrelated] = 4.8
    model.masks = [[left, unrelated]]; model.current = [dict(depth=depth)]; model.n = 1; model.evaluations = 0
    model.predicted = lambda *args: predicted
    model.cfg = dict(missing_support_penalty=12., occlusion_margin_m=.06, ray_depth_sigma_m=.06,
        huber_transition=2., boundary_sigma_px=1.5, change_penalty=4., motion_scale_m=.6,
        null_penalty=12., max_local_motion_m=.6, se3_translation_bound_m=3., se3_rotation_bound_deg=15.)
    return model


class KnownUnionDefectTests(unittest.TestCase):
    def test_unrelated_pixels_outside_history_support_are_invisible_to_factor(self):
        model = counterexample(); G = np.eye(4); delta = np.zeros((1, 3))
        correct = model.projective_factor(0, 0, [0, -1], key_matrix(G), delta)
        wrong = model.projective_factor(0, 0, [0, 0], key_matrix(G), delta)
        self.assertEqual(correct, wrong)
        self.assertEqual(wrong, (0., 32))

    def test_null_penalty_rewards_absorbing_unrelated_fragment(self):
        model = counterexample(); G = np.eye(4); delta = np.zeros((1, 3))
        rejected = model.energy('union_joint', G, delta, [[0, -1]])
        absorbed = model.energy('union_joint', G, delta, [[0, 0]])
        self.assertEqual(rejected-absorbed, 12.)


if __name__ == '__main__': unittest.main()
