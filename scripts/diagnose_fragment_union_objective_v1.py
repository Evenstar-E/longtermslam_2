"""POSTHOC cross-objective and positive-support diagnostics; no new reports."""
from pathlib import Path
import numpy as np
from inspection_decisions.fragment_union_joint_v1 import DirectFactors, key_matrix
from scripts.run_profiled_identity_actions_v1 import read, save, digest


def main():
    out = Path('results/fragment_union_joint_v1'); cfg = read(out/'frozen_contract.json')['config']
    rows = []; coverage = []
    for split in ['development', 'fresh']:
        for world in range(12):
            for action in cfg['acquisition_actions']:
                wd = out/split/f'{world:04d}'/f'a{action}'; raw = read(wd/'observed.json')
                result = read(wd/'backend.json'); metrics = read(wd/'evaluation.json'); model = DirectFactors(raw, cfg)
                union_output = result['outputs']['union_joint']; union_best = union_output['modes'][union_output['selected']]
                for method in ['point_one_to_one', 'point_many', 'point_set_ray', 'union_fixed_pose']:
                    output = result['outputs'][method]; state = output['modes'][output['selected']]
                    value = model.energy('union_joint', np.asarray(state['G']), np.asarray(state['delta']), state['assignment'])
                    rows.append(dict(split=split, world=world, action=action, state_from=method,
                        candidate_energy=union_best['score'], energy_at_other_state=value,
                        other_state_is_lower=value < union_best['score']-1e-7,
                        candidate_task_loss=metrics['union_joint']['task_loss'], other_task_loss=metrics[method]['task_loss'],
                        other_state_has_better_task=metrics[method]['task_loss'] < metrics['union_joint']['task_loss']-1e-9))
                state = union_best; Gkey = key_matrix(np.asarray(state['G'])); delta = np.asarray(state['delta'])
                identity = {r['ref_id']: r for r in metrics['union_joint']['ownership_records']}
                for f, labels in enumerate(state['assignment']):
                    for j in set(labels)-{-1}:
                        mask, _ = model.mask_view(f, tuple(i for i, a in enumerate(labels) if a == j))
                        known = np.isfinite(model.predicted(f, j, Gkey, tuple(delta[j])))
                        support_fraction = float((mask&known).sum()/max(int(mask.sum()), 1))
                        coverage.append(dict(split=split, world=world, action=action, frame=f, ref=j,
                            observed_pixels=int(mask.sum()), overlap_pixels=int((mask&known).sum()),
                            observed_explained_fraction=support_fraction,
                            identity_ok=identity[j]['identity_ok'], label=identity[j]['label']))
                model.predicted.cache_clear(); model.archive_points.cache_clear(); model.mask_view.cache_clear()
            print('CROSS OBJECTIVE', split, world, flush=True)
    aggregates = []
    for split in ['development', 'fresh']:
        for method in ['point_one_to_one', 'point_many', 'point_set_ray', 'union_fixed_pose']:
            subset = [r for r in rows if r['split'] == split and r['state_from'] == method]
            better = [r for r in subset if r['other_state_has_better_task']]
            aggregates.append(dict(split=split, state_from=method, endpoints=len(subset),
                other_better_task=len(better), lower_candidate_energy=sum(r['other_state_is_lower'] for r in subset),
                better_task_and_lower_candidate_energy=sum(r['other_state_is_lower'] for r in better),
                better_task_but_not_lower_candidate_energy=sum(not r['other_state_is_lower'] for r in better)))
    save(out/'objective_cross_diagnostic.json', dict(status='POSTHOC; all original predictions and method scores remain unchanged',
        source_sha256=digest(__file__), aggregates=aggregates, cross_objective_rows=rows, support_rows=coverage,
        interpretation='A lower energy at an existing ordinary solution proves a missed feasible state for this local solver. A higher value does not prove a globally optimal wrong model; it only compares these specific states.',
        support_caveat='Current pixels outside acquired archive support may be legitimate unseen surfaces. Low support alone is not a valid universal rejection rule; no threshold tuned here.'))
    print('CROSS OBJECTIVE', aggregates, flush=True)


if __name__ == '__main__': main()
