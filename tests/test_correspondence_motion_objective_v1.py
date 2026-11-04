import unittest
import numpy as np
from inspection_decisions.correspondence_motion_objective_v1 import solve,profile_cost,motion_fit,energy,frames_from
from scripts.run_evidence_value_pilot_v1 import read


def evidence(X,Ys):
    n=len(X)
    return dict(references=[dict(ref_id=i) for i in range(n)],frames=[dict(token=str(f),pixels=[[i] for i in range(len(Y))],
        D=(Y[:,None]-X[None]).tolist(),S=np.zeros((len(Y),n)).tolist(),valid=np.ones((len(Y),n),bool).tolist()) for f,Y in enumerate(Ys)])


class MotionTests(unittest.TestCase):
    def setUp(self):self.cfg=read('configs/correspondence_motion_objective_v1.json')
    def test_non_saturating_tail_and_real_large_motion_not_forbidden(self):
        c=[float(profile_cost(np.array([x,0,0]),.06,2,.6*np.sqrt(2))) for x in [.3,.9,1.8]]
        self.assertTrue(c[0]<c[1]<c[2])
        fitted=motion_fit([[.9,0,0],[.9,0,0]],np.zeros(3),self.cfg)
        self.assertLess(np.linalg.norm(fitted-[.9,0,0]),.01)
    def test_permuted_correspondence_shared_drift_and_change(self):
        X=np.array([[0,0,0],[2,0,0],[4,0,0],[0,2,0],[2,2,0],[4,2,0]],float)
        p=np.array([3,1,5,0,4,2]);Y=X[p]+[.9,0,0];Y[2]+=[0,.3,0]
        data=evidence(X,[Y,Y]);r=solve(data,'shared_motion',self.cfg)
        self.assertEqual(dict(r['best']['matches'][0]),dict(enumerate(p.tolist())))
        np.testing.assert_allclose(r['best']['translation'],[.9,0,0],atol=.003)
        self.assertLess(np.linalg.norm(np.array(r['best']['delta'][5])-[0,.3,0]),.003)
        self.assertLessEqual(max(np.diff(r['best']['energy_trace'])),1e-7)
    def test_exact_periodic_ambiguity_is_retained(self):
        self.cfg['public_session_translation_bound_m']=2.
        X=np.array([[0,0,0],[2,0,0],[4,0,0]],float)
        r=solve(evidence(X,[X[:2],X[:2]]),'shared_motion',self.cfg)
        self.assertGreaterEqual(len(r['branches']),2)
        self.assertTrue(all(a['label']==-1 for a in r['prediction']['assets']))
    def test_no_observation_is_unknown(self):
        r=solve(evidence(np.zeros((2,3)),[np.empty((0,3))]),'shared_motion',self.cfg)
        self.assertTrue(all(a['label']==-1 for a in r['prediction']['assets']))
    def test_conflicting_views_cannot_be_absorbed_by_one_shared_motion(self):
        data=evidence(np.zeros((1,3)),[np.array([[.3,0,0]]),np.array([[-.3,0,0]])])
        shared=energy(frames_from(data),[[(0,0)],[(0,0)]],np.zeros(3),np.zeros((1,3)),self.cfg)
        independent=2*float(profile_cost(np.array([.3,0,0]),.06,2,.6*np.sqrt(2)))
        self.assertGreater(shared,10*independent)


if __name__=='__main__':unittest.main()
