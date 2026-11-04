"""Post-seal surrogate diagnosis and public-input integrity checks; no inference."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read, save, sha
from scripts.prepare_real_temporal_v1 import read_ply


def main():
    old_root = Path('results/joint_source_regret_v1')
    new_root = Path('results/joint_source_balanced_v2')
    old = read(old_root / 'episode_02/predictions.json')
    new = read(new_root / 'episode_02/predictions.json')
    for root in [old_root, new_root]:
        for p, h in read(root / 'prediction_seal.json')['sha256'].items():
            assert sha(root / p) == h
    a = old['choices']['minimax_source_regret']
    b = new['choices']['minimax_source_regret']
    lo = np.array([r['loss']['deletion_losses'] for r in old['rows']])
    ln = np.array([r['loss']['deletion_losses'] for r in new['rows']])
    bo, bn = lo.min(axis=0), ln.min(axis=0)
    ar_old, ar_new = float(max(ln[a] - bo)), float(max(ln[a] - bn))
    br_old, br_new = float(max(ln[b] - bo)), float(max(ln[b] - bn))
    assert br_old < ar_old and br_new < ar_new
    assert int(np.argmax(np.abs(bo-bn))) == 0
    save(new_root / 'regret_failure_diagnosis.json', {
        'scope': 'Post-seal diagnosis of episode_02 only; no selection or parameter changes.',
        'old_choice': a, 'new_choice': b,
        'old_baselines': bo.tolist(), 'new_baselines': bn.tolist(),
        'old_choice_losses': ln[a].tolist(), 'new_choice_losses': ln[b].tolist(),
        'old_choice_regret_old_new': [ar_old, ar_new],
        'new_choice_regret_old_new': [br_old, br_new],
        'old_subset_choice_under_new_baselines': int(np.argmin(np.max(lo-bn, axis=1))),
        'interpretation': 'New state improves regret even under old baselines, but worsens reference anchor errors. Baseline shift alone is not the cause; surrogate tradeoff is misaligned with physical task.'})

    root = Path('data/rescan_third_capture_v1')
    for p, h in read(root / 'seal.json')['sha256'].items():
        assert sha(root / p) == h
    public = read(root / 'public.json')['clouds']
    raw = read(root / 'acquisition.json')['records']
    records = []
    for p, r in zip(public, raw):
        x, y = read_ply(root / r['path']), read_ply(root / p['path'])
        assert x.dtype == y.dtype and x.shape == y.shape
        fields = [f for f in x.dtype.names if f not in ['instance_idx', 'class_idx']]
        for f in fields:
            assert np.array_equal(x[f], y[f], equal_nan=True), f
        assert np.all(y['instance_idx'] == -1) and np.all(y['class_idx'] == 0)
        records.append({'scene': p['scene'], 'points': len(y), 'unchanged_fields': fields,
                        'labels_removed': True, 'same_order_and_geometry': True})
    save(root / 'public_integrity_audit.json', {
        'passed': True, 'records': records,
        'scope': 'Geometry/attribute preservation and removal of privileged labels only. No identity evaluation or calibration validation.',
        'script_sha256': sha(__file__)})
    print('PASS: regret counterexample and both third-capture public inputs')


if __name__ == '__main__':
    main()
