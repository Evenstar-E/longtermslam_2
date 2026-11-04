import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from comparative_evidence.interface import Episode, shortest_path
from comparative_evidence.scene import convex_depth, generate_scene, halfspaces
from comparative_evidence.policy import execute_policy


class EvidenceInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = Path('.cache/evidence_interface_tests').resolve()
        cache.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=cache)
        cls.root = Path(cls.temp.name)/'scene'
        cfg = json.loads(Path('configs/comparative_evidence_interface_v1.json').read_text())
        cfg.update(image_width=48, image_height=36, focal_px=42., minimum_component_pixels=3)
        cls.cfg = cfg
        generate_scene(cls.root, 1234, dict(limited_history=False, coherent_error=True,
                       repeated_appearance=True, changed_assets=3), cfg)

    @classmethod
    def tearDownClass(cls):
        assert Path(cls.temp.name).resolve().parent == Path('.cache/evidence_interface_tests').resolve()
        cls.temp.cleanup()

    def test_convex_renderer_intersection_and_miss(self):
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (4, 6)])
        planes = halfspaces([vertices])[0]
        d = convex_depth(np.zeros(3), np.array([[0., 0., 1.], [2., 0., 1.], [0., 0., -1.]]), planes)
        self.assertAlmostEqual(d[0], 4.)
        self.assertTrue(np.isinf(d[1:]).all())

    def test_budget_rejection_no_partial_mutation_and_cache_no_extra_charge(self):
        ep = Episode(self.root, 1, 1)
        a = ep.act('retrieve_history', 2)
        before = ep.state()
        a['depth'][:] = -999
        self.assertTrue((ep.act('retrieve_history', 2)['depth'] > 0).all())
        self.assertEqual(before, ep.state())
        with self.assertRaises(ValueError):
            ep.act('retrieve_history', 3)
        self.assertEqual(before, ep.state())
        for kind, view in (('truth', 0), ('observe_current', -1), ('observe_current', True)):
            with self.assertRaises(ValueError):
                ep.act(kind, view)
            self.assertEqual(before, ep.state())
        ep.act('observe_current', 3)
        before = ep.state()
        with self.assertRaises(ValueError):
            ep.act('observe_current', 4)
        self.assertEqual(before, ep.state())
        end = ep.finish()
        self.assertEqual(end['history_views'], 1)
        self.assertEqual(end['additional_current_views'], 1)
        self.assertGreater(end['travel_m'], 0)
        self.assertEqual(ep.finish(), end)
        with self.assertRaises(ValueError):
            ep.act('observe_current', 3)

    def test_truth_and_unexecuted_payload_not_read(self):
        # Remove evaluator truth and an unexecuted observation. Episode startup,
        # metadata, preview and other actions must remain usable.
        hidden = self.root/'payloads'/'current_11.npz'
        truth = self.root/'evaluation'/'truth.json'
        hidden_backup, truth_backup = hidden.with_suffix('.hold'), truth.with_suffix('.hold')
        hidden.rename(hidden_backup); truth.rename(truth_backup)
        try:
            ep = Episode(self.root, 1, 1)
            meta = ep.metadata(); preview = ep.preview()
            self.assertNotIn('rgb', meta['current'][11])
            self.assertNotIn('thumbnail_png_base64', meta['current'][11])
            self.assertNotIn('delta', meta)
            self.assertNotIn('owner', preview)
            self.assertEqual([e['view_id'] for e in ep.state()['events']], [0])
            ep.act('retrieve_history', 0)
            ep.act('observe_current', 1)
            ep.finish()
        finally:
            hidden_backup.rename(hidden); truth_backup.rename(truth)

    def test_metadata_and_preview_defensive_copies(self):
        ep = Episode(self.root, 1, 1)
        m = ep.metadata(); m['edges'].clear()
        p = ep.preview(); p['rgb'][:] = 0
        self.assertTrue(ep.metadata()['edges'])
        self.assertGreater(ep.preview()['rgb'].sum(), 0)

    def test_path_cost_is_geodesic_not_teleportation(self):
        edges = [dict(a=0, b=1, distance_m=2.), dict(a=1, b=2, distance_m=3.),
                 dict(a=0, b=2, distance_m=9.)]
        self.assertEqual(shortest_path(edges, 0, 2), (5., [0, 1, 2]))
        with self.assertRaises(ValueError):
            shortest_path(edges, 0, 9)

    def test_common_policy_entry_returns_cost_without_evaluator(self):
        def policy(port):
            self.assertFalse(hasattr(port, 'truth'))
            self.assertFalse(hasattr(port, '_root'))
            port.preview()
            port.act('retrieve_history', 1)
            port.act('observe_current', 2)
            return {'status': 'interface_trace_only'}
        result = execute_policy(self.root, 1, 1, policy)
        self.assertEqual(result['prediction']['status'], 'interface_trace_only')
        self.assertTrue(result['cost']['returned'])
        self.assertGreaterEqual(result['timings']['policy_wall_excluding_api_s'], 0)

    def test_reproducible_sensor_payload_and_non_cuboid_geometry(self):
        other = Path(self.temp.name)/'replay'
        generate_scene(other, 1234, dict(limited_history=False, coherent_error=True,
                       repeated_appearance=True, changed_assets=3), self.cfg)
        try:
            for token in ('history_00', 'current_00', 'current_11'):
                with np.load(self.root/'payloads'/f'{token}.npz') as a, np.load(other/'payloads'/f'{token}.npz') as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            t = json.loads((other/'evaluation'/'truth.json').read_text())
            self.assertEqual(set(t['kinds']), {'hex_prism', 'wedge', 'bracket'})
            self.assertEqual(sum(t['changed']), 3)
            v = np.array(t['historical_vertices'][0][0])
            self.assertEqual(len(v), 12)
            self.assertGreater(np.abs(np.array(t['rotations'][0])-np.eye(3)).sum(), .1)
        finally:
            import shutil
            assert other.resolve().parent == Path(self.temp.name).resolve()
            shutil.rmtree(other)


if __name__ == '__main__':
    unittest.main()
