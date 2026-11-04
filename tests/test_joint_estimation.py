import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import fit_rigid,transform,infer,simulate,evaluate

class JointTest(unittest.TestCase):
    def test_se3_rotation_translation_and_reflection(self):
        p=np.random.default_rng(9).normal(size=(20,3));T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.2,-.4,.1]).as_matrix();T[:3,3]=[1,2,-3]
        recovered=fit_rigid(p,transform(p,T));np.testing.assert_allclose(recovered,T,atol=1e-10);self.assertAlmostEqual(np.linalg.det(recovered[:3,:3]),1.)
    def test_inference_sees_no_truth_and_recovery(self):
        import json
        from pathlib import Path
        cfg=json.loads(Path('configs/joint_estimation_v1.json').read_text());cond=dict(pose_error='low',change_fraction=0.,repeated_appearance=False,missing_fraction=0.)
        obs,truth=simulate(cfg['simulation'],cond,39)
        self.assertFalse(any(k.startswith('truth') or k in ['assignment','changed','T'] for k in obs))
        pred=infer(obs,cfg['algorithm'],'joint_history',339);scores=evaluate(pred,truth,cfg['evaluation'])
        self.assertTrue(scores['pose_success']);self.assertEqual(scores['assignment_accuracy'],1.)
    def test_wrong_identity_changes_are_penalized(self):
        pred=dict(T=np.eye(4),assignment=np.array([1,0]),changed=np.array([True,False]))
        truth=dict(T=np.eye(4),assignment=np.array([0,1]),changed=np.array([True,False]))
        score=evaluate(pred,truth,dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=3))
        self.assertEqual((score['tp'],score['fp'],score['fn']),(0,1,1))
    def test_decoupled_recovers_pose_with_changed_landmark(self):
        from scripts.joint_estimation_v2 import infer_decoupled
        import json
        from pathlib import Path
        cfg=json.loads(Path('configs/joint_estimation_v1.json').read_text());cond=dict(pose_error='high',change_fraction=.3,repeated_appearance=False,missing_fraction=0.)
        obs,truth=simulate(cfg['simulation'],cond,52);pred=infer_decoupled(obs,cfg['algorithm'],339)
        score=evaluate(pred,truth,cfg['evaluation'])
        self.assertTrue(score['pose_success']);self.assertEqual(score['assignment_accuracy'],1.);self.assertGreater(score['f1'],.9)

if __name__=='__main__':unittest.main()
