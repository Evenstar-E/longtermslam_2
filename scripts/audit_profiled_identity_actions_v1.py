"""Independent API/evaluation replay and nonlinear check of forecast profiling."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.interface import Episode, shortest_path
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.estimation_v1 import reference_map
from comparative_evidence.scene import PALETTE
from inspection_decisions.projective_visibility_v1 import capture, decode, triangles_from_mask
from inspection_decisions.se3_visibility_v1 import meshes_from_history
from inspection_decisions.profiled_identity_actions_v1 import backend, forecast, choose, feature
from scripts.run_profiled_identity_actions_v1 import read, save, digest, identity_diagnostics


def perturbation(eta, cfg):
    P = np.eye(4); eta = np.asarray(eta)
    P[:3, 3] = eta[:3]*cfg['nuisance_translation_m']
    P[:3, :3] = Rotation.from_rotvec(eta[3:]*np.deg2rad(cfg['nuisance_rotation_deg'])).as_matrix()
    return P


def nonlinear_profile_check(raw, fitted, forecasts, metadata, cfg):
    hs = [decode(o) for o in raw['history']]
    meshes = meshes_from_history(hs, raw['sensor'], cfg)
    o = decode(raw['current'][0])
    color_distance = np.linalg.norm(o['rgb'].astype(float)[:, :, None]-PALETTE, axis=3).min(axis=2)
    foreground = triangles_from_mask(o['depth'], o['nominal_T'],
                                    (color_distance > 30)&(o['depth'] < 8.), raw['sensor'], cfg)
    sensor = dict(raw['sensor']); factor = cfg['forecast_downsample']
    sensor['image_width'] //= factor; sensor['image_height'] //= factor; sensor['focal_px'] /= factor
    rows = []
    for action in forecasts['actions']:
        T = np.asarray(metadata['current'][action['action']]['nominal_T'])
        for pair in action['profiles']:
            features = []
            for branch, eta in [(pair['k'], pair['eta'][:6]), (pair['l'], pair['eta'][6:])]:
                b = fitted['options'][branch]
                phi, _ = feature(meshes, np.asarray(b['map_transform_current_from_archive']),
                    np.asarray(b['best']['delta']), T, sensor, cfg, perturbation(eta, cfg), foreground)
                features.append(phi)
            actual = float(np.sum((features[0]-features[1])**2)+cfg['nuisance_ridge']*np.sum(np.asarray(pair['eta'])**2))
            rows.append(dict(action=action['action'], k=pair['k'], l=pair['l'], raw=pair['raw'],
                linear_profiled=pair['profiled'], actual_at_linear_eta=actual,
                linear_underestimate=actual-pair['profiled'], actual_exceeds_original=actual > pair['raw']+1e-9))
    return rows


def main(out=None):
    cfg = read('configs/profiled_identity_actions_v1.json'); out = Path(out or cfg['output'])
    frozen = read(out/'frozen_contract.json'); cfg = frozen['config']; checks = 0
    if frozen.get('frozen_before_repaired_replay'):
        from scripts.recover_profiled_identity_actions_v1 import install
        install()
    for path, sha in frozen['sources'].items():
        assert digest(path) == sha, path; checks += 1
    sensor = read('configs/comparative_evidence_interface_v1.json'); sensor['factors'] = {}
    summary = read(out/'summary.json'); rows = read(out/'rows.json'); check_rows = []; nonlinear = []
    backend_replays = []; raw_frame_count = 0
    for world in range(cfg['worlds']):
        wd = out/f'{world:04d}'; root = out/'scenes'/f'{world:04d}'
        catalog = read(root/'catalog.json'); truth = read(root/'evaluation/truth.json')
        assert truth['path_minimum_z_clearance_m'] > sensor['clearance_m']; checks += 1
        raw_frame_count += len(list((root/'payloads').glob('*.npz')))
        for action in range(7):
            ep = Episode(root, 1, 1); hs = [ep.act('retrieve_history', 0)]; qs = [ep.preview()]
            if action: qs.append(ep.act('observe_current', action))
            raw = capture(hs, qs, sensor)
            prefix = 'initial' if action == 0 else f'endpoint_{action}'
            assert raw == read(wd/f'{prefix}_observed.json'); checks += 1
            cached = read(wd/f'{prefix}_backend.json')
            selected = min(range(len(cached['options'])), key=lambda k: cached['options'][k]['best']['score'])
            assert cached['selected'] == selected and cached['prediction'] == cached['options'][selected]['prediction']; checks += 1
            metric = evaluate(root, cached['prediction'])
            assert metric == read(wd/f'endpoint_{action}_evaluation.json'); checks += 1
            cost = ep.finish(); saved_cost = read(wd/f'endpoint_{action}_cost.json')
            for key in ('travel_m', 'additional_current_views', 'modeled_acquisition_s', 'returned', 'payload_bytes'):
                assert cost[key] == saved_cost['cost'][key]; checks += 1
            assert [(e['action'], e.get('payload_sha256'), e['path'], e['travel_m']) for e in ep.state()['events']] == [
                (e['action'], e.get('payload_sha256'), e['path'], e['travel_m']) for e in saved_cost['events']]; checks += 1
            diag = identity_diagnostics(root, raw, cached)
            saved_diag = read(wd/('initial_identity_diagnostic.json' if action == 0 else f'endpoint_{action}_identity_diagnostic.json'))
            assert json.loads(json.dumps(diag)) == saved_diag; checks += 1
            for branch in cached['registration']['retained']:
                for pairs in branch['matches']:
                    assert len({i for i, j in pairs}) == len(pairs) == len({j for i, j in pairs}); checks += 1
            for record in catalog['current']:
                a = record['id']; P = np.linalg.inv(np.asarray(catalog['current'][0]['nominal_T']))@record['nominal_T']
                actual_P = np.linalg.inv(np.asarray(truth['poses']['current_00']))@truth['poses'][f'current_{a:02d}']
                np.testing.assert_allclose(P, actual_P, atol=1e-12); checks += 1
            # Two full backends replayed: an initial and an actual changed-task
            # endpoint. This is deterministic verification, not additional trials.
            if (world, action) in [(0, 0), (4, 1)]:
                replay = backend(raw, cfg)
                canonical = json.loads(json.dumps(replay, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))
                assert canonical == cached; checks += 1; backend_replays.append([world, action])
        initial = read(wd/'initial_observed.json'); fitted = read(wd/'initial_backend.json')
        forecasts = read(wd/'forecasts.json'); decisions = read(wd/'decisions.json')
        replay = forecast(initial, fitted, catalog, cfg)
        assert replay == forecasts; checks += 1
        for action in forecasts['actions']:
            raw_d = np.asarray(action['raw_distance']); profiled_d = np.asarray(action['profiled_distance'])
            assert np.all(profiled_d <= raw_d+1e-10) and np.min(profiled_d) >= 0; checks += 1
            for p in action['profiles']:
                assert max(abs(np.asarray(p['eta'])), default=0) <= 1+1e-9; checks += 1
                assert p['success']; checks += 1
        for method in cfg['methods']:
            a = choose(method, forecasts); assert a == decisions[method]; checks += 1
            row = next(r for r in rows if r['world'] == world and r['method'] == method)
            assert row['action'] == a and row['metric'] == read(wd/f'endpoint_{a}_evaluation.json'); checks += 1
        nl = nonlinear_profile_check(initial, fitted, forecasts, catalog, cfg)
        nonlinear += [dict(world=world, **r) for r in nl]
        check_rows.append(dict(world=world,
            candidate_action=decisions['profiled_report_risk'], ordinary_action=decisions['ordinary_report_risk'],
            actions_differ=decisions['profiled_report_risk'] != decisions['ordinary_report_risk']))
        print('AUDIT world', world, 'checks', checks, flush=True)
    assert raw_frame_count == summary['raw_RGBD_payloads']; checks += 1
    old = read('results/se3_visibility_v1/frozen_contract.json')
    # Prior main experiment source hashes are a read-only regression boundary.
    old_sources = old.get('sources', old.get('source_hashes', {}))
    assert old_sources, 'Missing old source inventory'
    for path, sha in old_sources.items():
        assert digest(path) == sha, path; checks += 1
    changed = [r for r in rows if r['changed'] and r['method'] == 'profiled_report_risk']
    nonlinear_summary = dict(pairs=len(nonlinear),
        actual_exceeds_original=sum(r['actual_exceeds_original'] for r in nonlinear),
        mean_linear_underestimate=float(np.mean([r['linear_underestimate'] for r in nonlinear])) if nonlinear else 0.)
    save(out/'nonlinear_profile_diagnostic.json', dict(summary=nonlinear_summary, rows=nonlinear,
        status='POSTHOC approximation audit; no scores or decisions modified; not nonlinear optimized minimum'))
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'audit.json'}
    save(out/'audit.json', dict(checks=checks, pass_all=True, API_endpoint_replays=7*cfg['worlds'],
        full_backend_replays=backend_replays, full_forecast_replays=cfg['worlds'],
        source_count=len(frozen['sources']), old_source_count=len(old_sources),
        candidate_vs_ordinary=check_rows, differing_action_worlds=sum(r['actions_differ'] for r in check_rows),
        nonlinear_profile=nonlinear_summary, artifacts=artifacts, audit_source_sha256=digest(__file__), exclusions=[]))
    print('AUDIT PASS', checks, 'nonlinear', nonlinear_summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output')
    main(parser.parse_args().output)
