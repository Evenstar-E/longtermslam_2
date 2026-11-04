import copy,unittest
import numpy as np
from inspection_decisions.report_risk_stopping_v1 import features,first_stop,fit_head,predict


def prediction():
    return dict(assets=[dict(ref_id=i,label=0,decision_risk=.02,delta=[0.,0.,0.],covariance=np.eye(3).tolist(),
                            current_support=20,association_margin_m=1.) for i in range(3)])


class RiskStopTests(unittest.TestCase):
    def test_temporal_and_permutation(self):
        p=prediction();q=copy.deepcopy(p);q['assets'][0]['delta'][0]=.3
        x=features(q,p,p,1,.2,2,True)
        for v in (q,p):v['assets'].reverse()
        np.testing.assert_array_equal(x,features(q,p,p,1,.2,2,True))
        static=features(q,None,q,1,.2,2,False)
        np.testing.assert_array_equal(x[:len(static)],static)
        self.assertFalse(np.array_equal(x,features(q,q,q,1,.2,2,True)))
    def test_lazy_stop_no_future_access(self):
        def states():
            yield dict(stage=0,prediction=prediction(),best_rank_score=-.1,best_coverage_score=1)
            raise AssertionError('Read future evidence after stop')
        self.assertEqual(first_stop(states(),'declared_rank',{},.05)[0],0)
        self.assertEqual(first_stop(states(),'original_rank',{},None)[0],0)
    def test_fit_train_only_deterministic(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(80,12));Y=np.c_[X[:,0]**2,X[:,1]-.2]
        a=fit_head(X,Y,.1);b=fit_head(X,Y,.1);self.assertEqual(a,b)
        self.assertTrue(np.isfinite(predict(X[0],a)).all())
        self.assertGreaterEqual(predict(X[0],a)[0],0)
    def test_budget_overrides_all_rules(self):
        s=dict(stage=5,prediction=prediction(),best_rank_score=10,best_coverage_score=1)
        self.assertEqual(first_stop(iter([s]),'dual_temporal',{},-1)[0],5)


if __name__=='__main__':unittest.main()
