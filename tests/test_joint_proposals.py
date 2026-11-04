import json,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import transform
from scripts.joint_estimation_v3 import solve as solve_v3
from scripts.joint_estimation_v4 import solve,coverage

class ProposalsTest(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text())['algorithm'];self.sampler=json.loads(Path('configs/joint_proposals_v4.json').read_text())
    def test_hard_sampler_is_v3_and_budget_fixed(self):
        a=np.load('results/joint_estimation_v1/scenes/0450.npz');obs={k:a[k] for k in a.files if not k.startswith('truth_')}
        prior=solve_v3(obs,self.cfg,44);got=solve(obs,self.cfg,self.sampler,44,'hard',True)
        for key in ['T','assignment','changed']:np.testing.assert_array_equal(prior[key],got[key])
        self.assertEqual(len(got['initial_poses']),25);self.assertEqual(len(got['solutions']),25)
    def test_geometry_recovers_anonymous_scalene_constellation(self):
        ref=np.array([[0,0,0],[1.1,.2,.1],[.4,1.7,.3],[.3,.5,2.2]])
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.3,-.2,.5]).as_matrix();T[:3,3]=[.7,1.,-.5]
        ids=np.array([2,0,3,1]);current=transform(ref[ids],np.linalg.inv(T));obs=dict(history_positions=ref[None],history_descriptors=np.zeros((1,4,8)),current_positions=current,current_descriptors=np.zeros((4,8)),initial_T=np.eye(4))
        pred=solve(obs,self.cfg,self.sampler,7,'geometry',True)
        np.testing.assert_allclose(pred['T'],T,atol=1e-8);np.testing.assert_array_equal(pred['assignment'],ids)
        cov=coverage(pred,dict(T=T),dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=3))
        self.assertTrue(cov['initial_coverage']);self.assertTrue(cov['final_coverage']);self.assertFalse(cov['selection_miss'])

if __name__=='__main__':unittest.main()
