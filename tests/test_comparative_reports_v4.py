import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.decisions_v4 import action_risks,report_dictionary,marginal_quadrature
from comparative_evidence.decisions_v4 import terminal_ablation
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information


class ReportsV4Tests(unittest.TestCase):
    def test_loss_matches_hand_calculation_and_superset(self):
        samples=np.array([[0,0,0],[.2,0,0],[.4,0,0]])
        w=np.array([.2,.3,.5]);reports=np.array([[.2,0,0],[.3,0,0]])
        risks=action_risks(reports,samples,w)
        np.testing.assert_allclose(risks[:3],[.8,.35,.66])
        base=report_dictionary(samples,np.array([False,True,True]))
        extended=report_dictionary(samples,np.array([False,True,True]),[[.27,0,0]])
        np.testing.assert_array_equal(base,extended[:-1])
        self.assertLessEqual(action_risks(extended,samples,w).min(),action_risks(base,samples,w).min())

    def test_quadrature_preserves_rare_mode_and_is_nested(self):
        modes=[dict(mean=np.array([x,0,0]),covariance=np.eye(3)*.001) for x in (0.,1.)]
        fit=dict(delta_start=0,modes=modes,mode_weights=[.9999,.0001])
        a,w=marginal_quadrature(fit,0,9);b,v=marginal_quadrature(fit,0,11)
        np.testing.assert_array_equal(a[:512],b[:512])
        np.testing.assert_array_equal(a[512:],b[2048:2560])
        self.assertAlmostEqual(w.sum(),1);self.assertAlmostEqual(w[512:].sum(),.0001)
        np.testing.assert_allclose(np.average(a,axis=0,weights=w),[.0001,0,0],atol=.001)

    def test_real_inference_without_truth_and_without_mutating_fit(self):
        original_text=Path.read_text;original_bytes=Path.read_bytes
        def read_text(p,*args,**kwargs):
            if 'evaluation' in p.parts:raise AssertionError('Truth text access')
            return original_text(p,*args,**kwargs)
        def read_bytes(p,*args,**kwargs):
            if 'evaluation' in p.parts:raise AssertionError('Truth bytes access')
            return original_bytes(p,*args,**kwargs)
        with patch.object(Path,'read_text',read_text),patch.object(Path,'read_bytes',read_bytes):
            ep=Episode(Path('results/comparative_interface_v1/scenes/0003'),2,1)
            h=[ep.act('retrieve_history',i) for i in (0,11)]
            q=[ep.preview(),ep.act('observe_current',1)]
            fit=attach_geometry_information(estimate_modes(h,q,ep.metadata()),h,q,ep.metadata())
            import copy
            before=copy.deepcopy(fit['assets']);predictions=terminal_ablation(fit)
            self.assertEqual(before,fit['assets'])
            for original,extended in (('legacy','centres_same32'),('qmc512','centres_qmc512')):
                for a,b in zip(predictions[original]['assets'],predictions[extended]['assets']):
                    self.assertLessEqual(b['decision_risk'],a['decision_risk']+1e-12)
                    self.assertEqual(a['point_estimate_delta'],b['point_estimate_delta'])
                    self.assertEqual(a['covariance'],b['covariance'])


if __name__=='__main__':unittest.main()
