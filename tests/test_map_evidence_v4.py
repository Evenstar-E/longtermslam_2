import unittest
import numpy as np
from scipy.optimize import linprog
from scripts.map_evidence_v4 import Model


class MapEvidenceTest(unittest.TestCase):
    def test_correlated_joint_set_can_exclude_null_while_box_contains_it(self):
        # x+y>=.30; coordinate ranges each [0,.30] intersect [-.12,.12],
        # but the joint feasible set cannot intersect the no-change square.
        A=np.array([[-1.,-1.]])
        self.assertTrue(linprog([1,0],A_ub=A,b_ub=[-.3],bounds=[(0,.3)]*2).success)
        self.assertFalse(linprog([0,0],A_ub=A,b_ub=[-.3],bounds=[(0,.12)]*2).success)

    def test_no_current_constraints_does_not_create_movement_evidence(self):
        cfg=dict(motion_bound_m=.7,motion_axis_threshold_m=.12,box_width_prior_m=[.3,1.5],max_changed_assets=1,measurement_tolerance_m=.025,milp_time_limit_s=3)
        inp=dict(assets=1,frames=1,assignment=[0],history=[],current=[])
        m=Model(inp,cfg);r=m.solve();self.assertEqual(r.status,0)
        obj=np.zeros(m.size);obj[m.d0]=-1;r=m.solve(obj)
        self.assertGreater(r.x[m.d0],.12)


if __name__=='__main__':unittest.main()
