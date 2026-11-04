import copy,unittest
import numpy as np
from inspection_decisions.optional_stopping_v1 import fit_risk,ordinary_bank,crossfit_bank,value,first_stop
from inspection_decisions.report_risk_stopping_v1 import feature_row


def data():
    return [dict(family=i,policy='rank',X=np.eye(6),L=np.array([.9,.8,.7,.6,0.,1.])) for i in range(8)]


class OptionalTests(unittest.TestCase):
    def test_optional_future_stop_changes_value(self):
        rows=data();risk=fit_risk(rows,1e-8)
        recursive,_=ordinary_bank(rows,risk,1e-8,.1)
        tail,_=ordinary_bank(rows,risk,1e-8,.1,'full_tail')
        self.assertAlmostEqual(value(rows[0]['X'][3],recursive[3]),.1,places=5)
        self.assertAlmostEqual(value(rows[0]['X'][3],tail[3]),1.2,places=5)
        # At t3 risk .6: recursive says continue to t4 then stop; full tail says stop now.
        self.assertLess(value(rows[0]['X'][3],recursive[3]),.6)
        self.assertGreater(value(rows[0]['X'][3],tail[3]),.6)
    def test_fold_targets_exclude_held_family_labels(self):
        rows=data();risk=fit_risk(rows,1.)
        _,a=crossfit_bank(rows,risk,1.,.01,4)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['family']%4==0:r['L']+=10
        _,b=crossfit_bank(changed,fit_risk(changed,1.),1.,.01,4)
        ids=[i for i,r in enumerate(rows) if r['family']%4==0]
        np.testing.assert_array_equal(np.array(a['targets'])[ids,:4],np.array(b['targets'])[ids,:4])
        self.assertTrue((np.array(b['targets'])[ids,4]>np.array(a['targets'])[ids,4]).all())
        for teacher in a['teachers']:
            self.assertFalse(set(teacher['training_families'])&set(teacher['held_families']))
    def test_lazy_stop_does_not_read_future(self):
        p=dict(assets=[dict(ref_id=0,label=0,decision_risk=.02,delta=[0,0,0],covariance=np.eye(3).tolist(),current_support=20,association_margin_m=1.)])
        s=dict(stage=0,prediction=p,best_rank_score=.1,best_coverage_score=1.)
        n=len(feature_row(s,None,p,False))
        def constant(v):return dict(mean=[0.]*n,scale=[1.]*n,w=[[v,0.]]+[[0.,0.]]*n,train_min=[-1e6]*n,train_max=[1e6]*n)
        head=dict(temporal=False,risk=constant(.02),bank=[constant(.03)]*5)
        bundle=dict(heads={'optional_cf_temporal':head})
        def states():
            yield s
            raise AssertionError('Future state accessed')
        self.assertEqual(first_stop(states(),'optional_cf_temporal',bundle,.05)[0],0)
        s['stage']=5
        self.assertEqual(first_stop(iter([s]),'optional_cf_temporal',{},-1)[0],5)
    def test_deterministic_targets_and_terminal_cost(self):
        rows=data();risk=fit_risk(rows,1.)
        a,detail=crossfit_bank(rows,risk,1.,.01,4)
        b,again=crossfit_bank(rows,risk,1.,.01,4)
        self.assertEqual(a,b);self.assertEqual(detail,again)
        np.testing.assert_allclose(np.array(detail['targets'])[:,-1],1.01)


if __name__=='__main__':unittest.main()
