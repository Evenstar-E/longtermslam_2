import unittest,copy
from unittest.mock import patch
import numpy as np
from inspection_decisions.component_conflict_v1 import fit,predict,representation

class ComponentTests(unittest.TestCase):
    def fixture(self):
        p=dict(assets=[dict(ref_id=j,label=0,decision_risk=.02+j*.1,delta=[0,0,0],covariance=(np.eye(3)*.001).tolist(),current_support=20,association_margin_m=1.) for j in range(2)])
        e=dict(weights=[.5,.5,0,0],mode_deltas=[[[0,0,0],[0,0,0]],[[.2,0,0],[0,0,0]],[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]],selected=[[0,0,0],[0,0,0]],sensor_var=[.001,.001],observations={'0':[],'1':[]})
        return dict(stage=0,prediction=p,evidence=e,best_rank_score=.1,best_coverage_score=1.)
    def test_binding_and_permutation(self):
        s=self.fixture();ids,X=representation(s,None,s['prediction'],'component_conflict')
        self.assertFalse(np.array_equal(X[0,-12:],X[1,-12:]))
        _,unpaired=representation(s,None,s['prediction'],'component_unpaired')
        np.testing.assert_array_equal(unpaired[0,-12:],unpaired[1,-12:])
        s['prediction']['assets'].reverse()
        ii,XX=representation(s,None,s['prediction'],'component_conflict')
        self.assertEqual(ids,ii);np.testing.assert_array_equal(X,XX)
    def test_no_filesystem_and_modal_ablation(self):
        s=self.fixture()
        with patch('builtins.open',side_effect=AssertionError('No filesystem in representation')):
            _,x=representation(s,None,s['prediction'],'component_no_modes')
        np.testing.assert_array_equal(x[:,-12:-4],0)
    def test_state_weighting_and_order(self):
        X=np.array([[1.,2.],[3,4],[2,1]]);y=np.array([0.,1.,.35]);w=np.array([.5,.5,1.])
        m=fit(X,y,w,10);n=fit(X[::-1],y[::-1],w[::-1],10)
        np.testing.assert_allclose(predict(X,m),predict(X,n))
        repeated=fit(np.repeat(X,2,0),np.repeat(y,2),np.repeat(w/2,2),10)
        np.testing.assert_allclose(predict(X,m),predict(X,repeated))
    def test_nonnegative_finite(self):
        m=fit([[1,1],[1,1]],[0,1],[1,1],10)
        self.assertTrue(np.isfinite(predict(np.array([[1,1]]),m)).all())
        self.assertGreaterEqual(predict(np.array([[1,1]]),m)[0],0)

if __name__=='__main__':unittest.main()
