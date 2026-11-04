import unittest
import numpy as np
from scripts.roi_change_baseline import compare,anchor_component,interval_present,query_map

class ReprojectionTest(unittest.TestCase):
    def setUp(self):
        self.d=np.ones((12,12))*2
        self.mask=np.ones((12,12),bool)
        self.K=np.array([[10.,0,6],[0,10,6],[0,0,1]])
        self.labels=np.ones((12,12),int)*6
        self.cfg=dict(minimum_projected_pixels=32,minimum_depth_m=.1,minimum_projected_fraction=.25,depth_tolerance_m=.05,minimum_valid_fraction=.5,decision_fraction=.6)
    def run_compare(self,depth,T=None,method='occlusion_aware'):
        return compare(self.d,depth,self.mask,self.K,np.eye(4) if T is None else T,self.labels,6,self.cfg,method)
    def test_rigid_consistent_and_farther(self):
        T=np.eye(4);T[2,3]=.2
        self.assertEqual(self.run_compare(self.d+.2,T)['result'],'consistent')
        self.assertEqual(self.run_compare(self.d+1)['result'],'unsupported')
    def test_near_occluder_is_not_change(self):
        self.assertEqual(self.run_compare(self.d-1)['result'],'insufficient')
        self.assertEqual(self.run_compare(self.d-1,method='naive')['result'],'unsupported')
    def test_outside_and_invalid_reject(self):
        T=np.eye(4);T[0,3]=100
        self.assertEqual(self.run_compare(self.d,T)['reason'],'projection_coverage')
        self.assertEqual(self.run_compare(self.d*np.nan)['reason'],'invalid_depth')
    def test_mask_uses_observed_component(self):
        labels=np.zeros((12,12),int);labels[1:4,1:4]=6;labels[7:9,7:9]=6
        mask=anchor_component(self.d,labels,[0,0,11,11],6)
        self.assertEqual(int(mask.sum()),9)
        self.assertTrue(mask[2,2]);self.assertFalse(mask[8,8])
    def test_interval_and_missing_map_class(self):
        self.assertTrue(interval_present([0],[2**64-1],100))
        self.assertFalse(interval_present([1],[3],3))
        self.assertTrue(interval_present([1,5],[3],5))
        self.assertEqual(query_map([],np.zeros((1,3)),6,1,2,{})['result'],'insufficient')

if __name__=='__main__':unittest.main()
