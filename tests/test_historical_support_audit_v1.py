import unittest
from types import SimpleNamespace

import numpy as np

from scripts.audit_historical_support_v1 import supported_faces, fix_supported_reference, score


class HistoricalSupportAuditTest(unittest.TestCase):
    def test_only_observation_supported_faces_are_fixed(self):
        inp = {'history': [{'asset': 0, 'endpoints': [
            {'axis': 0, 'side': 0, 'tight': True},
            {'axis': 0, 'side': 1, 'tight': False}]}]}
        model = SimpleNamespace(lower=np.full(6, -20.), upper=np.full(6, 20.))
        truth = {'reference_boxes': [[[-2., -3., -4.], [2., 3., 4.]]]}
        self.assertEqual(supported_faces(inp['history']), {(0, 0, 0)})
        fix_supported_reference(model, inp, truth)
        self.assertEqual(model.lower[0], -2.)
        self.assertEqual(model.upper[0], -2.)
        np.testing.assert_array_equal(model.lower[1:], -20.)
        np.testing.assert_array_equal(model.upper[1:], 20.)

    def test_unknown_and_wrong_identity_do_not_become_success(self):
        s = score([-1, 1, 0], {'assignment': [0, 1, 2]},
                  {'changed': [True, True, False], 'assignment': [0, 9, 2]})
        self.assertEqual(s, dict(visible=3, changed=2, known=2, tp=0, fp=1, wrong_known=1))


if __name__ == '__main__':
    unittest.main()
