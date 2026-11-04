import copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.support_forecast_v5 import prepare_support,reciprocal_forecast
from comparative_evidence.selection_v5 import common_losses,common_terminal,run_selection_v5
from comparative_evidence.geometry_modes_v2 import mixture_particles
from comparative_evidence.decisions_v4 import terminal_ablation
from comparative_evidence.policy import execute_policy


class SelectionV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ep=Episode(Path('results/comparative_interface_v1/scenes/0003'),2,1)
        cls.h=[ep.act('retrieve_history',0)];cls.q=[ep.preview()];cls.meta=ep.metadata()
        cls.fit=attach_geometry_information(estimate_modes(cls.h,cls.q,cls.meta),cls.h,cls.q,cls.meta)

    def test_paired_model_cancels_shape_and_preserves_calibration(self):
        cache=prepare_support(self.fit,self.h,self.q,self.meta)
        for epoch in ('history','current'):
            future,a=reciprocal_forecast((epoch,1),self.fit,self.meta,cache)
            self.assertIsNotNone(future);B,R,_=future
            np.testing.assert_array_equal(B[:,:self.fit['delta_start']],0)
            self.assertGreater(np.linalg.norm(B[:,self.fit['nuisance_start']:]),0)
            self.assertGreater(np.linalg.eigvalsh(R).min(),0)
            self.assertGreater(np.max(np.abs(R-np.diag(np.diag(R)))),0)
            self.assertGreater(a['opposite_visible_points'],0)
            self.assertGreaterEqual(a['mean_missing_support'],0)
            self.assertLessEqual(a['mean_missing_support'],1+1e-12)

    def test_terminal_matches_frozen_v4_and_does_not_mutate_fit(self):
        before=copy.deepcopy(self.fit['assets']);p=common_terminal(self.fit)
        prior=terminal_ablation(self.fit)['centres_qmc2048']
        self.assertEqual(before,self.fit['assets'])
        for a,b in zip(p['assets'],prior['assets']):
            for key in ('delta','label','decision_risk','point_estimate_delta','covariance'):
                self.assertEqual(a[key],b[key])

    def test_common_regret_is_symmetric_and_nonnegative(self):
        losses,regret=common_losses(self.fit,mixture_particles(self.fit))
        np.testing.assert_allclose(regret,regret.T)
        np.testing.assert_allclose(np.diag(regret),0,atol=1e-12)
        self.assertGreaterEqual(regret.min(),0)
        for L in losses:self.assertEqual(L.shape[1],32)

    def test_actual_acquisition_has_no_truth_or_hidden_payload_access(self):
        original_text=Path.read_text;original_bytes=Path.read_bytes
        def read_text(p,*args,**kwargs):
            if 'evaluation' in p.parts:raise AssertionError('Truth text')
            return original_text(p,*args,**kwargs)
        def read_bytes(p,*args,**kwargs):
            if 'evaluation' in p.parts:raise AssertionError('Truth bytes')
            return original_bytes(p,*args,**kwargs)
        with patch.object(Path,'read_text',read_text),patch.object(Path,'read_bytes',read_bytes):
            result=execute_policy(Path('results/comparative_interface_v1/scenes/0003'),2,1,
                                  lambda port:run_selection_v5(port,'support_candidate'))
        self.assertEqual(result['cost']['history_views'],2)
        self.assertEqual(result['cost']['additional_current_views'],1)
        self.assertTrue(result['cost']['returned'])
        self.assertEqual(len(result['prediction']['decisions']),2)
        # Actual reads only preview, anchor and one legal action per remaining slot.
        self.assertEqual(len([e for e in result['state']['events'] if e['payload_bytes']>0]),4)
        for d in result['prediction']['decisions']:
            for a in d['alternatives']:
                for token in a['source_tokens']:
                    self.assertIn(token,('history_00','current_00','history_'+f"{result['prediction']['decisions'][0]['view']:02d}",
                        'current_'+f"{result['prediction']['decisions'][0]['view']:02d}"))


if __name__=='__main__':unittest.main()
