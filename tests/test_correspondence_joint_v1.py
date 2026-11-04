import unittest
import numpy as np
from inspection_decisions.correspondence_joint_v1 import solve,report
from scripts.run_evidence_value_pilot_v1 import read

class JointTests(unittest.TestCase):
    def test_partial_periodic_scene_is_not_uniquely_identifiable(self):
        X=np.array([[0,0,0],[2,0,0],[4,0,0]],float);Y=X[:2]
        D=Y[:,None]-X[None]
        data=dict(references=[dict(ref_id=i) for i in range(3)],frames=[dict(token='q',pixels=[[0],[1]],D=D.tolist(),S=np.zeros((2,3)).tolist(),valid=np.ones((2,3),bool).tolist())])
        cfg=read('configs/correspondence_joint_pilot_v1.json');cfg['public_session_translation_bound_m']=2.
        result=solve(data,'joint_branches',cfg)
        self.assertGreaterEqual(len(result['branches']),2)
        self.assertTrue(all(a['label']==-1 for a in result['prediction']['assets']))
    def test_real_assignments_shared_drift(self):
        X=np.array([[0,0,0],[2,0,0],[4,0,0],[0,2,0],[2,2,0],[4,2,0]],float);perm=np.array([3,1,5,0,4,2]);Y=X[perm]+[.9,0,0];Y[2]+=[0,.3,0]
        D=Y[:,None]-X[None];n=len(X);data=dict(references=[dict(ref_id=j,anchor_token='h',anchor_pixels=[j]) for j in range(n)],frames=[dict(token='q',pixels=[[i] for i in range(n)],D=D.tolist(),S=np.zeros((n,n)).tolist(),valid=np.ones((n,n),bool).tolist())])
        result=solve(data,'joint_branches',read('configs/correspondence_joint_pilot_v1.json'))
        np.testing.assert_allclose(result['best']['translation'],[.9,0,0],atol=1e-8)
        self.assertEqual(dict(result['best']['matches'][0]),dict(enumerate(perm.tolist())))
        self.assertGreater(result['distinct_assignment_hypotheses'],1)
    def test_disagreeing_branches_unknown(self):
        data=dict(references=[dict(ref_id=i) for i in range(2)],frames=[dict(token='q',pixels=[[0],[1]])])
        a=dict(matches=[[(0,0),(1,1)]],delta=[[0,0,0],[0,0,0]]);b=dict(matches=[[(0,1),(1,0)]],delta=[[0,0,0],[0,0,0]])
        p=report(data,a,[a,b]);self.assertTrue(all(x['label']==-1 for x in p['assets']))
    def test_empty_observation_unknown(self):
        data=dict(references=[dict(ref_id=0)],frames=[dict(token='q',pixels=[],D=[],S=[],valid=[])])
        p=solve(data,'joint_branches',read('configs/correspondence_joint_pilot_v1.json'))
        self.assertEqual(p['prediction']['assets'][0]['label'],-1)

if __name__=='__main__':unittest.main()
