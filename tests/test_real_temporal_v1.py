import unittest
import numpy as np
from inspection_decisions.real_temporal_v1 import assign,fit_stats,moved

class RealTemporalTests(unittest.TestCase):
    def test_landmark_displacement_not_translation(self):
        T=np.eye(4);T[:2,:2]=[[0,-1],[1,0]];c=np.array([[2.,0,0]])
        np.testing.assert_allclose(moved(c,T)-c,[[-2,2,0]])
        self.assertGreater(np.linalg.norm((moved(c,T)-c)[0]-T[:3,3]),1)
    def test_assignment_rejects_unexplained_asset(self):
        result=assign(np.array([[.1,.9],[.8,.85]]),{'null_assignment_cost':.65})
        self.assertEqual(result,{0:0,1:None})
    def test_correspondence_column_permutation(self):
        a=np.array([[.1,.9],[.8,.2]])
        first=assign(a,{'null_assignment_cost':.65});second=assign(a[:,::-1],{'null_assignment_cost':.65})
        self.assertEqual(first,{i:1-j for i,j in second.items()})
    def test_surface_fit_does_not_prove_material_identity(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(100,3))
        self.assertEqual(fit_stats(x,x,np.eye(4))['cost'],0)
        # A physically distinct copy has the same geometric score; scorer must retain identity labels.
        self.assertEqual(fit_stats(x.copy(),x,np.eye(4))['cost'],0)

if __name__=='__main__':unittest.main()
