"""Matched-input direct-inference experiment; dev and fresh seeds kept separate."""
import argparse
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.fragment_union_joint_v1 import run_methods
from scripts.run_profiled_identity_actions_v1 import read, save, digest, generate, acquired

CFG = 'configs/fragment_union_joint_v1.json'
SOURCES = [CFG, __file__, 'research/fragment_union_joint_contract_v1_2026-10-23.md',
    'inspection_decisions/fragment_union_joint_v1.py', 'tests/test_fragment_union_joint_v1.py',
    'inspection_decisions/stable_branch_readout_v1.py', 'inspection_decisions/se3_visibility_v1.py',
    'inspection_decisions/projective_visibility_v1.py', 'inspection_decisions/correspondence_joint_v1.py',
    'inspection_decisions/correspondence_motion_objective_v1.py', 'inspection_decisions/profiled_identity_actions_v1.py',
    'comparative_evidence/estimation_v1.py', 'comparative_evidence/interface.py', 'comparative_evidence/scene.py',
    'comparative_evidence/evaluation_v1.py', 'comparative_evidence/evaluation_v3.py',
    'scripts/partial_surface_scene.py', 'scripts/joint_estimation.py',
    'scripts/run_profiled_identity_actions_v1.py', 'configs/profiled_identity_actions_v1.json',
    'configs/se3_visibility_v1.json', 'configs/comparative_evidence_interface_v1.json']


def config():
    cfg = read('configs/se3_visibility_v1.json')
    cfg.update(read('configs/profiled_identity_actions_v1.json')); cfg.update(read(CFG))
    return cfg


def mode_metric(G, truth):
    G = np.asarray(G); true_G = np.asarray(truth['session_G'])
    return dict(rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(G[:3, :3]@true_G[:3, :3].T).magnitude())),
                translation_error_m=float(np.linalg.norm(G[:3, 3]-true_G[:3, 3])))


def summarize(rows):
    result = []
    for split in sorted({r['split'] for r in rows}):
        for action in [0, 1, 6]:
            for method in sorted({r['method'] for r in rows}):
                subset = [r for r in rows if r['split'] == split and r['method'] == method and (action == 0 or r['action'] == action)]
                if not subset: continue
                result.append(dict(split=split, action='pooled_correlated' if action == 0 else action, method=method,
                    endpoints=len(subset), unique_worlds=len({r['world'] for r in subset}),
                    qualified_tp=sum(r['metric']['qualified_tp'] for r in subset),
                    true_changes=sum(r['metric']['changed'] for r in subset),
                    fp=sum(r['metric']['fp'] for r in subset),
                    bad_motion=sum(r['metric']['bad_motion_reports'] for r in subset),
                    wrong_known=sum(r['metric']['wrong_known'] for r in subset),
                    known=sum(r['metric']['known'] for r in subset),
                    mean_task_loss=float(np.mean([r['metric']['task_loss'] for r in subset])),
                    mean_method_s=float(np.mean([r['method_elapsed_s'] for r in subset])),
                    mean_travel_m=float(np.mean([r['cost']['travel_m'] for r in subset]))))
    return result


def execute(command):
    cfg = config(); smoke = command == 'smoke'
    out = Path('results/fragment_union_joint_smoke_v1' if smoke else cfg['output'])
    assert not out.exists(), 'Output already sealed'; out.mkdir(parents=True)
    sources = {str(p): digest(p) for p in SOURCES}
    save(out/'frozen_contract.json', dict(config=cfg, sources=sources, smoke=smoke,
        frozen_before_this_execution=True, fresh_seeds_not_generated_before_freeze=not smoke,
        scope='known development smoke only' if smoke else 'old dev plus same-family fresh exploratory worlds; no novelty confirmation'))
    sensor = read('configs/comparative_evidence_interface_v1.json'); sensor['factors'] = {}
    rows = []; paths = 0; started = time.perf_counter()
    splits = [('development', 1 if smoke else cfg['development_worlds'])]
    if not smoke: splits.append(('fresh', cfg['fresh_worlds']))
    for split, worlds in splits:
        for world in range(worlds):
            if split == 'development':
                root = Path(cfg['development_source'])/f'{world:04d}'
            else:
                root = out/'scenes'/'fresh'/f'{world:04d}'; generate(root, world, cfg, sensor)
            actions = [6] if smoke else cfg['acquisition_actions']
            for action in actions:
                wd = out/split/f'{world:04d}'/f'a{action}'; wd.mkdir(parents=True)
                ep, raw = acquired(root, action, sensor); paths += 1
                save(wd/'observed.json', raw)
                result = run_methods(raw, cfg); save(wd/'backend.json', result)
                cost = ep.finish(); save(wd/'cost.json', dict(cost=cost, events=ep.state()['events']))
                # All method outputs persist before any truth/owner evaluation.
                truth = read(root/'evaluation/truth.json'); metrics = {}
                for method, output in result['outputs'].items():
                    metric = evaluate(root, output['prediction']); metrics[method] = metric
                    pose = None
                    if method != 'legacy_compressed': pose = mode_metric(output['modes'][output['selected']]['G'], truth)
                    row = dict(split=split, world=world, action=action, method=method,
                        metric=metric, cost=cost, method_elapsed_s=output['elapsed_s'], pose_metric=pose,
                        evaluations=output.get('evaluations', 0), render_calls=output.get('render_calls', 0))
                    rows.append(row)
                save(wd/'evaluation.json', metrics)
                save(out/'progress.json', dict(split=split, world=world, action=action,
                    paths=paths, rows=len(rows), elapsed_s=time.perf_counter()-started))
                print(split, world, 'a', action, 'modes', len(result['registration']['retained']),
                    {m:(v['qualified_tp'], v['fp'], round(v['task_loss'], 3)) for m, v in metrics.items()}, flush=True)
    assert all(digest(p) == sha for p, sha in sources.items())
    save(out/'rows.json', rows)
    save(out/'summary.json', dict(rows=summarize(rows), API_paths=paths, backend_calls=paths,
        method_outputs=len(rows), new_worlds=0 if smoke else cfg['fresh_worlds'],
        fresh_stored_payloads=0 if smoke else 8*cfg['fresh_worlds'],
        fresh_acquired_unique_payloads=0 if smoke else 4*cfg['fresh_worlds'],
        elapsed_s=time.perf_counter()-started, source_hashes_unchanged=True,
        fixed_acquisition=True, novelty_confirmed=False, exclusions=[]))
    print('DONE', command, 'paths', paths, 'rows', len(rows), 'seconds', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['smoke', 'run'])
    execute(parser.parse_args().command)
