import unittest
import numpy as np
from scripts.shared_pose_change_v3 import choose_changed,robust_energy


class SharedPoseTest(unittest.TestCase):
    def test_small_gain_unchanged_and_minimum_stable_gauge(self):
        cfg=dict(motion_penalty_m2=.01,minimum_stable_objects=3)
        pc,g=choose_changed([.20,.02,.03,.01,.4],[0.,0.,0.,0.,0.],cfg)
        np.testing.assert_array_equal(pc,[True,False,False,False,True])
        self.assertEqual((~pc).sum(),3)

    def test_robust_energy_matches_soft_l1_definition(self):
        self.assertEqual(robust_energy(np.zeros(5),.1),0)
        self.assertAlmostEqual(robust_energy(np.array([.1]),.1),.02*(np.sqrt(2)-1))


if __name__=='__main__':unittest.main()
