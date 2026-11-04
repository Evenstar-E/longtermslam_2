"""Independent data-contract audit; no task-success or novelty estimate."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from comparative_evidence.interface import shortest_path


def main():
    root = Path('results/comparative_interface_v1')
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = json.loads((root/'frozen_config.json').read_text())
    rows = [json.loads(s) for s in (root/'episodes.jsonl').read_text().splitlines()]
    checks = 0
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        checks += 1
    assert len(rows) == 144 and manifest['completed_scenes'] == 48 and not manifest['exclusions']
    checks += 1
    census = dict(history_preview_visible_assets=0, current_preview_visible_assets=0,
                  moved_assets=0, current_preview_visible_moved_assets=0)
    cells = Counter()
    for scene in manifest['scenes']:
        sid = scene['scene']; case = root/'scenes'/f'{sid:04d}'
        catalog = json.loads((case/'catalog.json').read_text())
        truth = json.loads((case/'evaluation'/'truth.json').read_text())
        cells[json.dumps(scene['condition'], sort_keys=True)] += 1
        assert len(catalog['history']) == len(catalog['current']) == 12
        assert set(truth['kinds']) == {'hex_prism', 'wedge', 'bracket'}
        assert len(set(truth['groups'])) == (4 if scene['condition']['repeated_appearance'] else 8)
        assert sum(truth['changed']) == scene['condition']['changed_assets']
        delta = np.asarray(truth['delta'])
        assert np.array_equal(np.max(np.abs(delta), axis=1) > .12, truth['changed'])
        checks += 5
        for name, digest in scene['hashes'].items():
            assert hashlib.sha256((case/name).read_bytes()).hexdigest() == digest, (sid, name)
            checks += 1
        for epoch in ('history', 'current'):
            for entry in catalog[epoch]:
                token = entry['token']
                with np.load(case/'payloads'/f'{token}.npz', allow_pickle=False) as data:
                    assert set(data.files) == {'rgb', 'depth', 'nominal_T'}
                    assert np.isfinite(data['depth']).all() and (data['depth'] > 0).all()
                    assert data['rgb'].shape == (96, 128, 3) and data['depth'].shape == (96, 128)
                    np.testing.assert_array_equal(data['nominal_T'], entry['nominal_T'])
                u = np.array(truth['realized_nuisance'][token]); T = np.array(truth['poses'][token])
                nominal = np.array(entry['nominal_T'])
                assert np.abs(u).max() <= 1
                np.testing.assert_allclose(nominal[:3, 3]-T[:3, 3], u[:3]*cfg['translation_bound_m'], atol=1e-12)
                expected = Rotation.from_rotvec(u[3:6]*cfg['rotation_bound_rad']).as_matrix() @ T[:3, :3]
                np.testing.assert_allclose(nominal[:3, :3], expected, atol=1e-12)
                assert 'thumbnail_png_base64' in entry if epoch == 'history' else 'thumbnail_png_base64' not in entry
                assert entry['nuisance_group'] == (epoch if scene['condition']['coherent_error'] else token)
                checks += 9
        hvisible = np.asarray(scene['preview_visible_pixel_counts']['history_00']) >= cfg['minimum_component_pixels']
        cvisible = np.asarray(scene['preview_visible_pixel_counts']['current_00']) >= cfg['minimum_component_pixels']
        census['history_preview_visible_assets'] += int(hvisible.sum())
        census['current_preview_visible_assets'] += int(cvisible.sum())
        census['moved_assets'] += sum(truth['changed'])
        census['current_preview_visible_moved_assets'] += int((cvisible & truth['changed']).sum())
        episodes = [r for r in rows if r['scene'] == sid]
        assert len(episodes) == 3
        checks += 1
        for row in episodes:
            state, cost = row['state'], row['cost']; events = state['events']
            assert cost['closed'] and cost['returned'] and state['position'] == 0
            assert cost['history_views'] == state['limits']['history']
            assert cost['additional_current_views'] == state['limits']['current']
            assert [e['sequence'] for e in events] == list(range(len(events)))
            position = 0; travel = 0; read_tokens = []
            for event in events:
                if event['action'] in ('preview', 'retrieve_history', 'observe_current'):
                    token = f"{event['epoch']}_{event['view_id']:02d}"
                    raw = (case/'payloads'/f'{token}.npz').read_bytes()
                    assert event['payload_bytes'] == len(raw)
                    assert event['payload_sha256'] == hashlib.sha256(raw).hexdigest()
                    read_tokens.append(token); checks += 2
                if event['action'] in ('observe_current', 'return_to_preview'):
                    goal = event['view_id'] if event['action'] == 'observe_current' else 0
                    distance, path = shortest_path(catalog['edges'], position, goal)
                    assert abs(distance-event['travel_m']) < 1e-10 and path == event['path']
                    position = goal; checks += 1
                travel += event['travel_m']
            assert len(set(read_tokens)) == len(read_tokens)
            assert len(read_tokens) == 1+cost['history_views']+cost['additional_current_views']
            assert abs(travel-cost['travel_m']) < 1e-10
            assert cost['index_bytes'] == len((case/'catalog.json').read_bytes())
            assert abs(cost['modeled_acquisition_s']-(1+cost['additional_current_views'])*cfg['acquisition_s']) < 1e-10
            checks += 9
    assert len(cells) == 16 and set(cells.values()) == {3}
    checks += 1
    summary = dict(stage='Engineering/interface validation only, no algorithm comparison',
                   checks=checks, scenes=48, sensor_payloads=48*24, action_traces=len(rows),
                   factorial_cells=len(cells), repeats_per_cell=3, exclusions=[],
                   preview_census_evaluator_only=census,
                   source_and_all_data_hashes_unchanged=True,
                   total_files_bytes=sum(p.stat().st_size for p in (root/'scenes').rglob('*') if p.is_file()))
    (root/'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
