import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from comparative_evidence.geometry_modes_v2 import estimate_modes,audit_energy
from comparative_evidence.estimation_local_v2 import estimate as local_v2
from comparative_evidence.estimation_v1 import estimate as local_v1
from comparative_evidence.interface import Episode
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v2 import task_action_losses,decision_risk,run_selection_v2


ROOT=Path('results/comparative_interface_v1/scenes/0003')


class SelectionV2Tests(unittest.TestCase):
    def test_explicit_start_copy_preserves_default_local_solver(self):
        e=Episode(ROOT,2,1)
        h=[e.act('retrieve_history',i) for i in (0,11)]
        q=[e.preview(),e.act('observe_current',1)]
        old=local_v1(h,q,e.metadata());new=local_v2(h,q,e.metadata())
        np.testing.assert_array_equal(old['mean'],new['mean'])
        np.testing.assert_array_equal(old['covariance'],new['covariance'])
        self.assertEqual(old['assets'],new['assets'])

    def test_tolerance_compatible_reports_not_falsely_disagreed(self):
        d=np.array([[[.25,0,0]],[[.37,0,0]]]);changed=np.ones((2,1),bool)
        losses,regret=task_action_losses(d,changed)
        self.assertAlmostEqual(regret[0,1],0)
        self.assertAlmostEqual(decision_risk(np.array([.5,.5]),losses),0)
        # Once no .10m ball can cover both motions, there is real conflict.
        d[1,0,0]=.48
        losses,regret=task_action_losses(d,changed)
        self.assertGreater(regret[0,1],0)
        self.assertGreater(decision_risk(np.array([.5,.5]),losses),0)

    def test_common_geometry_energy_and_representative_weights(self):
        e=Episode(ROOT,4,2)
        h=[e.act('retrieve_history',i) for i in (0,1,10,11)]
        q=[e.preview()]+[e.act('observe_current',i) for i in (1,10)]
        f=estimate_modes(h,q,e.metadata())
        energies=[m['energy']['total'] for m in f['modes']]
        chosen=next(i for i,m in enumerate(f['mode_audit']) if m['selected'])
        self.assertAlmostEqual(energies[chosen],min(energies))
        self.assertAlmostEqual(f['mode_weights'].sum(),1)
        self.assertTrue((f['mode_weights']>=0).all())
        self.assertAlmostEqual(audit_energy(f,h,q,e.metadata())['total'],min(energies))
        self.assertGreater(np.linalg.eigvalsh(f['covariance']).min(),0)

    def test_candidate_and_risk_control_cannot_read_evaluator(self):
        original_text=Path.read_text;original_bytes=Path.read_bytes;original_load=np.load
        def text_guard(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth read')
            return original_text(p,*a,**k)
        def bytes_guard(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth bytes')
            return original_bytes(p,*a,**k)
        def load_guard(p,*a,**k):
            if isinstance(p,(str,Path)) and 'evaluation' in Path(p).parts:raise AssertionError('Truth array')
            return original_load(p,*a,**k)
        for method in ('candidate_regret','task_risk_compatible_mc'):
            with patch.object(Path,'read_text',text_guard),patch.object(Path,'read_bytes',bytes_guard),patch.object(np,'load',load_guard):
                result=execute_policy(ROOT,2,1,lambda port:run_selection_v2(port,method))
            self.assertEqual(result['cost']['history_views'],2)
            self.assertEqual(result['cost']['additional_current_views'],1)
            self.assertEqual(len(result['prediction']['mode_audit']),4)


if __name__=='__main__':unittest.main()
