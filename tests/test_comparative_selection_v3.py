import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information,information_at_state,perturb_icp_delta_covariance
from comparative_evidence.decisions_v3 import terminal_prediction
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v3 import run_selection_v3

ROOT=Path('results/comparative_interface_v1/scenes/0003')


class SelectionV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ep=Episode(ROOT,2,1);cls.h=[ep.act('retrieve_history',i) for i in (0,11)]
        cls.q=[ep.preview(),ep.act('observe_current',1)];cls.meta=ep.metadata()
        cls.old=estimate_modes(cls.h,cls.q,cls.meta)
        cls.gn=attach_geometry_information(cls.old,cls.h,cls.q,cls.meta)

    def test_information_matches_actual_geometry_without_changing_point(self):
        np.testing.assert_array_equal(self.old['mean'],self.gn['mean'])
        np.testing.assert_array_equal(self.old['mode_weights'],self.gn['mode_weights'])
        for m,a in zip(self.gn['modes'],self.gn['uncertainty_audit']):
            self.assertAlmostEqual(m['energy']['data'],a['data_energy'],places=7)
            self.assertGreater(a['active_points'],0)

    def test_nuisance_marginalization_does_not_reduce_conditional_variance(self):
        C,S,a=information_at_state(self.old['modes'][3],self.h,self.q,self.meta)
        k=len(S);difference=C[:k,:k]-S
        self.assertGreaterEqual(np.linalg.eigvalsh(difference).min(),-1e-9)

    def test_terminal_variants_are_independent_and_do_not_mutate_fit(self):
        fit=copy.deepcopy(self.gn);before=copy.deepcopy(fit['assets'])
        risk=terminal_prediction(fit,'risk');interval=terminal_prediction(fit,'two_sigma')
        self.assertEqual(fit['assets'],before)
        risk['assets'][0]['delta'][0]=999
        self.assertNotEqual(interval['assets'][0]['delta'][0],999)
        self.assertEqual(fit['assets'],before)
        for a in terminal_prediction(fit,'risk')['assets']:
            self.assertLessEqual(a['decision_risk'],.35+1e-12)

    def test_structured_perturbation_rebuilds_observation_map(self):
        C,a=perturb_icp_delta_covariance(self.gn,self.h,self.q,self.meta,draws=4)
        self.assertEqual(a['failed_draws'],0)
        self.assertEqual(C.shape,(3*len(self.gn['references']),)*2)
        self.assertGreater(np.linalg.eigvalsh(C).min(),0)

    def test_active_policy_cannot_read_evaluation_truth(self):
        original=Path.read_text;original_bytes=Path.read_bytes;original_load=np.load
        def text(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth text')
            return original(p,*a,**k)
        def raw(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth bytes')
            return original_bytes(p,*a,**k)
        def load(p,*a,**k):
            if isinstance(p,(str,Path)) and 'evaluation' in Path(p).parts:raise AssertionError('Truth array')
            return original_load(p,*a,**k)
        with patch.object(Path,'read_text',text),patch.object(Path,'read_bytes',raw),patch.object(np,'load',load):
            result=execute_policy(ROOT,2,1,lambda port:run_selection_v3(port,'candidate_regret'))
        self.assertEqual(result['cost']['additional_current_views'],1)
        self.assertEqual(result['prediction']['terminal_rule'],'risk')


if __name__=='__main__':unittest.main()
