"""Evaluator-only visibility counts, not attainable-success or policy forecasts."""
from pathlib import Path
import numpy as np
from scripts.run_profiled_identity_actions_v1 import read, save, digest


def main():
    out = Path('results/fragment_union_joint_v1'); cfg = read(out/'frozen_contract.json')['config']; rows = []
    for split in ['development', 'fresh']:
        for world in range(12):
            root = Path(cfg['development_source'])/f'{world:04d}' if split == 'development' else out/'scenes/fresh'/f'{world:04d}'
            truth = read(root/'evaluation/truth.json'); changed = set(np.flatnonzero(truth['changed']).tolist())
            for action in cfg['acquisition_actions']:
                raw = read(out/split/f'{world:04d}'/f'a{action}'/'observed.json')
                counts = np.zeros(8, int)
                for obs in raw['current']:
                    with np.load(root/'evaluation'/f"{obs['token']}_owner.npz") as z:
                        owner = z['owner'].ravel()
                    counts += np.bincount(owner[owner >= 0], minlength=8)
                visible = set(np.flatnonzero(counts >= cfg['minimum_report_pixels']).tolist())
                rows.append(dict(split=split, world=world, action=action, current_pixel_counts=counts.tolist(),
                    changed=sorted(changed), changed_visible=sorted(changed&visible),
                    changed_below_threshold=sorted(changed-visible)))
    summary = []
    for split in ['development', 'fresh']:
        for action in cfg['acquisition_actions']:
            r = [r for r in rows if r['split'] == split and r['action'] == action]
            summary.append(dict(split=split, action=action, changed_instances=sum(len(v['changed']) for v in r),
                changed_with_at_least_12_acquired_pixels=sum(len(v['changed_visible']) for v in r),
                changed_below_12_acquired_pixels=sum(len(v['changed_below_threshold']) for v in r)))
    save(out/'visibility_diagnostic.json', dict(status='POSTHOC evaluator-only diagnostic; no inference input',
        source_sha256=digest(__file__), summary=summary, rows=rows,
        limits='Observed owner-pixel counts do not guarantee detected masks, correct identity, reconstructable displacement or an information-theoretic success bound.'))
    print('VISIBILITY', summary, flush=True)


if __name__ == '__main__': main()
