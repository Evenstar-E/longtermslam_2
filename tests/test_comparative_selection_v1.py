import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from comparative_evidence.estimation_v1 import estimate
from comparative_evidence.evaluation_v1 import score_prediction
from comparative_evidence.interface import Episode
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v1 import forecast, run_selection, task_risk


ROOT = Path('results/comparative_interface_v1/scenes/0001')


class SelectionTests(unittest.TestCase):
    def test_forecast_keeps_shared_frame_noise_correlation(self):
        ep = Episode(ROOT, 2, 1)
        h = [ep.act('retrieve_history', 0), ep.act('retrieve_history', 11)]
        fit = estimate(h, [ep.preview()], ep.metadata())
        B, R, coverage = forecast(('current', 1), fit, ep.metadata(),
                                 dict(history=[0, 11], current=[0]))
        off_diagonal = R-np.diag(np.diag(R))
        self.assertGreater(np.max(np.abs(off_diagonal)), 1e-7)
        self.assertGreater(np.linalg.eigvalsh(R).min(), 0)
        self.assertEqual(B.shape[1], len(fit['mean']))

    def test_generic_task_risk_rewards_a_correct_resolving_observation(self):
        changed = np.array([[False], [False], [True], [True]])
        close = np.eye(4, dtype=bool)[:, :, None]
        before = task_risk(np.ones(4)/4, changed, close)
        after_unchanged = task_risk(np.array([.5, .5, 0, 0]), changed, close)
        after_exact_change = task_risk(np.array([0, 0, 1., 0]), changed, close)
        self.assertGreater(before, 0)
        self.assertEqual(after_unchanged, 0)
        self.assertEqual(after_exact_change, 0)

    def test_candidate_runs_with_evaluation_reads_forbidden(self):
        original = Path.read_text
        original_bytes = Path.read_bytes
        original_load = np.load
        def guarded(path, *args, **kwargs):
            if 'evaluation' in path.parts:
                raise AssertionError('Truth access during policy')
            return original(path, *args, **kwargs)
        def guarded_bytes(path, *args, **kwargs):
            if 'evaluation' in path.parts:
                raise AssertionError('Evaluation bytes during policy')
            return original_bytes(path, *args, **kwargs)
        def guarded_load(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and 'evaluation' in Path(file).parts:
                raise AssertionError('Evaluation array during policy')
            return original_load(file, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded), patch.object(Path, 'read_bytes', guarded_bytes), patch.object(np, 'load', guarded_load):
            result = execute_policy(ROOT, 2, 1, lambda port: run_selection(port, 'candidate_separation'))
        self.assertEqual(result['cost']['history_views'], 2)
        self.assertEqual(result['cost']['additional_current_views'], 1)
        self.assertTrue(result['prediction']['references'])
        for a in result['prediction']['assets']:
            self.assertTrue(np.isfinite(a['delta']).all())

    def test_evaluator_does_not_reward_wrong_ownership_or_motion(self):
        cache = Path('.cache/selection_tests').resolve(); cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            base = Path(tmp); (base/'evaluation').mkdir()
            truth = dict(changed=[True]+[False]*7, delta=[[.3, 0, 0]]+[[0, 0, 0]]*7)
            (base/'evaluation'/'truth.json').write_text(json.dumps(truth))
            np.savez(base/'evaluation'/'history_00_owner.npz', owner=np.array([[0, 1]]))
            np.savez(base/'evaluation'/'current_00_owner.npz', owner=np.array([[0, 1]]))
            p = dict(references=[dict(ref_id=0, group=0, anchor_token='history_00', anchor_pixels=[0])],
                     assets=[dict(ref_id=0, label=1, delta=[.3, 0, 0])],
                     current_assignments=[dict(ref_id=0, token='current_00', pixels=[1])])
            wrong = score_prediction(base, p)
            self.assertEqual((wrong['tp'], wrong['qualified_tp'], wrong['fp']), (0, 0, 1))
            p['current_assignments'][0]['pixels'] = [0]
            p['assets'][0]['delta'] = [0, 0, 0]
            inaccurate = score_prediction(base, p)
            self.assertEqual((inaccurate['tp'], inaccurate['qualified_tp']), (1, 0))
            p['assets'][0]['label'] = -1
            unknown = score_prediction(base, p)
            self.assertEqual((unknown['changed'], unknown['tp'], unknown['fn']), (1, 0, 1))
            # TemporaryDirectory scope is checked before its recursive cleanup.
            self.assertEqual(base.resolve().parent, cache)


if __name__ == '__main__':
    unittest.main()
