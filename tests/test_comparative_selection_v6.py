import copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.support_forecast_v5 import prepare_support,reciprocal_forecast
from comparative_evidence.selection_v6 import uniform_support,forecast_variant,run_selection_v6
from comparative_evidence.selection_v5 import run_selection_v5
from comparative_evidence.policy import execute_policy


class SelectionV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ep=Episode(Path('results/comparative_interface_v1/scenes/0003'),2,1)
        h=[ep.act('retrieve_history',0)];q=[ep.preview()];cls.meta=ep.metadata()
        cls.fit=attach_geometry_information(estimate_modes(h,q,cls.meta),h,q,cls.meta)
        cls.cache=prepare_support(cls.fit,h,q,cls.meta)

    def test_uniform_intervention_only_replaces_weights_and_not_input(self):
        before=copy.deepcopy(self.cache);unit=uniform_support(self.cache)
        for epoch in self.cache:
            for p,q,b in zip(self.cache[epoch],unit[epoch],before[epoch]):
                for k in p:
                    if isinstance(p[k],np.ndarray):np.testing.assert_array_equal(p[k],b[k])
                    if k.startswith('missing_'):np.testing.assert_array_equal(q[k],np.ones_like(p[k]))
                    elif isinstance(p[k],np.ndarray):self.assertIs(p[k],q[k])
        actual,a=forecast_variant(('history',1),self.fit,self.meta,self.cache,unit,True)
        expected,_=reciprocal_forecast(('history',1),self.fit,self.meta,unit)
        for x,y in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(x,y)
        self.assertLess(a['mean_missing_support'],1);self.assertEqual(a['mean_pair_weight'],1)

    def test_weighted_loop_exactly_matches_v5_with_truth_guard(self):
        original_text=Path.read_text;original_bytes=Path.read_bytes
        def text(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth text')
            return original_text(p,*a,**k)
        def raw(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth bytes')
            return original_bytes(p,*a,**k)
        root=Path('results/comparative_interface_v1/scenes/0003')
        with patch.object(Path,'read_text',text),patch.object(Path,'read_bytes',raw):
            old=execute_policy(root,2,1,lambda port:run_selection_v5(port,'support_candidate'))
            result=execute_policy(root,2,1,lambda port:run_selection_v6(port,'weighted_candidate',old['prediction']['decisions']))
        self.assertEqual(old['prediction']['assets'],result['prediction']['assets'])
        self.assertEqual(len(result['prediction']['snapshots']),3)
        self.assertEqual(result['prediction']['snapshots'][-1]['prediction']['assets'],result['prediction']['assets'])
        self.assertTrue(result['cost']['returned'])


if __name__=='__main__':unittest.main()
