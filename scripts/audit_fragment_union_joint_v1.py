"""API, same-objective endpoint, report and deterministic full-solver replays."""
import copy
import json
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.fragment_union_joint_v1 import DirectFactors, run_methods, prediction
from scripts.run_profiled_identity_actions_v1 import read, save, digest, acquired


def canonical(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))


def without_timing(value):
    value = canonical(value); value.pop('total_elapsed_s', None)
    for output in value['outputs'].values(): output.pop('elapsed_s', None)
    return value


def main():
    out = Path('results/fragment_union_joint_v1'); frozen = read(out/'frozen_contract.json'); cfg = frozen['config']
    checks = 0; energy_states = 0; full_replays = []; API_paths = 0; grouping_rows = []
    for source, sha in frozen['sources'].items():
        assert digest(source) == sha, source; checks += 1
    sensor = read('configs/comparative_evidence_interface_v1.json'); sensor['factors'] = {}
    rows = read(out/'rows.json'); summary = read(out/'summary.json')
    for split, worlds in [('development', cfg['development_worlds']), ('fresh', cfg['fresh_worlds'])]:
        for world in range(worlds):
            root = Path(cfg['development_source'])/f'{world:04d}' if split == 'development' else out/'scenes/fresh'/f'{world:04d}'
            for action in cfg['acquisition_actions']:
                wd = out/split/f'{world:04d}'/f'a{action}'
                ep, raw = acquired(root, action, sensor); API_paths += 1
                assert raw == read(wd/'observed.json'); checks += 1
                cost = ep.finish(); saved_cost = read(wd/'cost.json')['cost']
                for key in ('travel_m', 'modeled_acquisition_s', 'additional_current_views', 'payload_bytes', 'returned'):
                    assert cost[key] == saved_cost[key]; checks += 1
                result = read(wd/'backend.json'); stored_metrics = read(wd/'evaluation.json'); model = DirectFactors(raw, cfg)
                assert [len(cs) for cs in model.frames] == result['acquired_partition_counts']; checks += 1
                for method, output in result['outputs'].items():
                    metric = evaluate(root, output['prediction'])
                    assert metric == stored_metrics[method]; checks += 1
                    row = next(r for r in rows if (r['split'], r['world'], r['action'], r['method']) == (split, world, action, method))
                    assert row['metric'] == metric; checks += 1
                    if method == 'legacy_compressed': continue
                    assert output['selected'] == min(range(len(output['modes'])), key=lambda k: output['modes'][k]['score']); checks += 1
                    for mode in output['modes']:
                        G, delta = np.asarray(mode['G']), np.asarray(mode['delta'])
                        value = model.energy(method, G, delta, mode['assignment'])
                        np.testing.assert_allclose(value, mode['score'], atol=1e-7, rtol=1e-10); checks += 1; energy_states += 1
                        assert np.max(np.diff(mode['trace']), initial=0) <= 0; checks += 1
                        assert mode['score'] <= mode['initial_score']+1e-9; checks += 1
                        if method == 'union_fixed_pose':
                            np.testing.assert_array_equal(G, mode['initial_G']); checks += 1
                        for f, labels in enumerate(mode['assignment']):
                            assert len(labels) == len(model.frames[f]); checks += 1
                            assert all(-1 <= j < model.n for j in labels); checks += 1
                            if method == 'point_one_to_one':
                                used = [j for j in labels if j >= 0]
                                assert len(set(used)) == len(used); checks += 1
                    pred = prediction(model, output['modes'][output['selected']], method)
                    assert canonical(pred) == output['prediction']; checks += 1
                    assert output['prediction']['delta_frame'] == 'archive'; checks += 1
                    best = output['modes'][output['selected']]
                    grouping_rows.append(dict(split=split, world=world, action=action, method=method,
                        repeated_reference_assignments=sum(len([j for j in labels if j >= 0])-len({j for j in labels if j >= 0})
                                                           for labels in best['assignment'])))
                    # Pixel ownership remains a partition of acquired samples.
                    for o in model.current:
                        pixels = [p for a in pred['current_assignments'] if a['token'] == o['token'] for p in a['pixels']]
                        assert len(pixels) == len(set(pixels)); checks += 1
                if (split, world, action) in [('fresh', 0, 6), ('fresh', 7, 1)]:
                    replay = run_methods(raw, cfg)
                    assert without_timing(replay) == without_timing(result); checks += 1
                    full_replays.append([split, world, action])
                model.predicted.cache_clear(); model.archive_points.cache_clear(); model.mask_view.cache_clear()
            print('AUDIT', split, world, 'checks', checks, flush=True)
    assert API_paths == summary['API_paths'] == 48 and len(rows) == 288; checks += 1
    assert summary['new_worlds'] == 12 and summary['fresh_stored_payloads'] == 96; checks += 1
    assert len(list((out/'scenes/fresh').glob('*/payloads/*.npz'))) == 96; checks += 1
    old = read('results/se3_visibility_v1/frozen_contract.json')
    for source, sha in old['sources'].items():
        assert digest(source) == sha, source; checks += 1
    save(out/'grouping_audit.json', dict(rows=grouping_rows, status='observed assignment multiplicity; not ground-truth identity success'))
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'audit.json'}
    save(out/'audit.json', dict(pass_all=True, checks=checks, API_replays=API_paths,
        method_evaluation_replays=288, same_objective_mode_recomputations=energy_states,
        full_backend_replays=full_replays, old_sources_unchanged=True, sources_unchanged=True,
        source_sha256=digest(__file__), artifacts=artifacts, exclusions=[]))
    print('AUDIT PASS', checks, 'energy modes', energy_states, flush=True)


if __name__ == '__main__': main()
