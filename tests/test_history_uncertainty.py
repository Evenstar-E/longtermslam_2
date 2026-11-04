import unittest
import numpy as np
from scripts.joint_estimation_v5 import fuse_history,skew

class HistoryTest(unittest.TestCase):
    def test_precision_fusion_and_covariance(self):
        positions=np.array([[[0.,0,0]],[[10.,0,0]]]);C=np.array([[np.eye(3)],[np.eye(3)*100]])
        mean,cov=fuse_history(positions,C)
        self.assertAlmostEqual(mean[0,0],10/101);np.testing.assert_allclose(cov[0],np.eye(3)*100/101)
        self.assertTrue((np.linalg.eigvalsh(cov)>0).all())
    def test_pose_covariance_rigid_translation_and_rotation_jacobian(self):
        p=np.array([1.,2.,3.]);omega=np.array([.01,-.02,.03]);np.testing.assert_allclose(-skew(p)@omega,np.cross(omega,p))
        J=np.column_stack([np.eye(3),-skew(p)]);C=J@np.diag([.01]*3+[.001]*3)@J.T
        self.assertTrue((np.linalg.eigvalsh(C)>0).all())
    def test_common_frame_bias_is_not_observable_absolute_pose(self):
        import json
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from scripts.joint_estimation import transform,evaluate
        from scripts.joint_estimation_v5 import solve
        cfg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text());sampler=json.loads(Path('configs/joint_proposals_v4.json').read_text())
        points=np.random.default_rng(15).normal(size=(8,3));E=np.eye(4);E[:3,:3]=Rotation.from_rotvec([.1,-.1,.2]).as_matrix();E[:3,3]=[.5,-.2,.1]
        obs=dict(history_positions=np.repeat(transform(points,E)[None],6,axis=0),history_descriptors=np.repeat(np.eye(8)[None],6,axis=0),history_covariances=np.tile(np.eye(3)*.0004,(6,8,1,1)),current_positions=points,current_descriptors=np.eye(8),current_variance_m2=.0004,initial_T=np.eye(4))
        pred=solve(obs,cfg['algorithm'],sampler,47);np.testing.assert_allclose(pred['T'],E,atol=1e-8)
        truth=dict(T=np.eye(4),assignment=np.arange(8),changed=np.zeros(8,bool))
        self.assertFalse(evaluate(pred,truth,cfg['evaluation'])['pose_success']);truth['T']=E
        self.assertTrue(evaluate(pred,truth,cfg['evaluation'])['pose_success']);self.assertFalse(pred['changed'].any())

if __name__=='__main__':unittest.main()
