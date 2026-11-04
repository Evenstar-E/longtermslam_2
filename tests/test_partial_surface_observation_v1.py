import copy
import unittest
import numpy as np
from inspection_decisions.partial_surface_observation_v1 import plane_shift,solve_directional,apply_surface_gate,bounded_quadratic
from scripts.run_evidence_value_pilot_v1 import read


class SurfaceTests(unittest.TestCase):
    def setUp(self):self.cfg=read('configs/partial_surface_observation_v1.json')
    def test_partial_plane_retains_tangent_nullspace(self):
        p=np.array([[x,y,0] for x in np.linspace(-1,1,15) for y in np.linspace(-1,1,15)])
        q=p[p[:,0]>.2]+[0,0,.25];pn=np.tile([0.,0.,1.],(len(p),1));qn=np.tile([0.,0.,1.],(len(q),1))
        d,s,H=plane_shift(p,pn,np.ones(len(p),bool),q,qn,np.ones(len(q),bool),self.cfg)
        self.assertAlmostEqual(d[2],.25,places=8)
        self.assertAlmostEqual(H[0,0],0);self.assertAlmostEqual(H[1,1],0)
        self.assertGreater(abs(d[0]),.2) # tangential median bias exists, but carries ZERO information.
    def test_complementary_views_ignore_unobserved_coordinate_bias(self):
        n=6;t=np.array([.2,.1,.3]);delta=np.zeros((n,3));delta[0]=[.3,.2,.1]
        data=dict(references=[dict(ref_id=j) for j in range(n)],frames=[])
        for f,H0 in enumerate([np.diag([0,0,3.]),np.diag([1.5,1.5,0])]):
            D=np.zeros((n,n,3));H=np.zeros((n,n,3,3));V=np.eye(n,dtype=bool)
            for j in range(n):D[j,j]=t+delta[j];H[j,j]=H0/self.cfg['sigma']**2
            data['frames'].append(dict(token=str(f),pixels=[[j] for j in range(n)],D=D.tolist(),S=np.zeros((n,n)).tolist(),valid=V.tolist(),H=H.tolist()))
        shifted=copy.deepcopy(data)
        for j in range(n):
            shifted['frames'][0]['D'][j][j][0]+=.7;shifted['frames'][0]['D'][j][j][1]-=.4
            shifted['frames'][1]['D'][j][j][2]-=.6
        a=solve_directional(data,self.cfg);b=solve_directional(shifted,self.cfg)
        np.testing.assert_allclose(a['best']['translation'],b['best']['translation'],atol=1e-5)
        np.testing.assert_allclose(a['best']['delta'],b['best']['delta'],atol=1e-5)
        self.assertLess(np.linalg.norm(np.array(b['best']['delta'][0])-delta[0]),.01)
        self.assertLessEqual(max(np.diff(b['best']['energy_trace'])),1e-6)
    def test_rank_deficient_surface_reports_unknown_for_both_models(self):
        data=dict(references=[dict(ref_id=0)],frames=[dict(token='q',pixels=[[0]],D=[[[0,0,0]]],S=[[0]],valid=[[True]],H=[[[[0,0,0],[0,0,0],[0,0,100]]]])])
        r=apply_surface_gate(solve_directional(data,self.cfg),data,self.cfg)
        self.assertEqual(r['prediction']['assets'][0]['label'],-1)
    def test_no_valid_normals_is_not_a_precise_measurement(self):
        p=np.zeros((10,3));N=np.tile([0,0,1.],(10,1));V=np.zeros(10,bool)
        self.assertIsNone(plane_shift(p,N,V,p,N,V,self.cfg))
    def test_empty_surface_evidence_is_unknown(self):
        data=dict(references=[dict(ref_id=0)],frames=[dict(token='q',pixels=[],D=[],S=[],valid=[],H=[])])
        r=solve_directional(data,self.cfg)
        self.assertEqual(r['prediction']['assets'][0]['label'],-1)
    def test_quadratic_bound_with_correlated_directions(self):
        H=np.array([[2.,1.,0],[1,2,0],[0,0,1]]);g=np.array([4.,0.,0.]);start=np.zeros(3)
        t=bounded_quadratic(H,g,start,1.)
        np.testing.assert_allclose(t,[1.,-.5,0],atol=1e-5)


if __name__=='__main__':unittest.main()
