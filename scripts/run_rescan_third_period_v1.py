"""Fresh capture2 native runs, with explicit initial/inherited-unfused contracts."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scripts.run_rescan_author_v1 import linux, sha
from scripts.prepare_real_temporal_v1 import read_ply


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2), encoding='utf-8')


def win(p):
    return Path(p.replace('/mnt/c/', 'C:/'))


def database_inputs(db):
    lines = db.read_text(encoding='utf-8').splitlines()
    folder = win(next(x.split(maxsplit=1)[1] for x in lines if x.startswith('model_folder ')))
    files = [db]
    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == 'object':
            files.append(folder / t[1])
        if t[0] == 'scene':
            files.append(win(t[3]))
            if t[4] != 'none':
                files.append(win(t[4]))
    return sorted(set(files))


def run(scene, mode, cfg, config, output):
    out = output / scene / mode
    out.mkdir(parents=True, exist_ok=False)
    prior = Path(cfg['prior_roots'][scene])
    db = prior / ('history.rsdb' if mode == 'initial_archive' else 'prediction.rsdb')
    current = Path(cfg['public_root']) / scene / 'capture2.ply'
    cloud = read_ply(current)
    assert np.all(cloud['instance_idx'] == -1) and np.all(cloud['class_idx'] == 0)
    inputs = database_inputs(db) + [current]
    binaries = Path(cfg['binaries'])
    sources = [Path(__file__), config, Path('scripts/run_rescan_author_v1.py'),
               Path('scripts/prepare_real_temporal_v1.py')] + [binaries / n for n in ['pose_proposal', 'segment_transfer']]
    manifest = {'scene': scene, 'mode': mode, 'configuration': cfg,
                'inputs': {str(p): sha(p) for p in inputs},
                'sources': {str(p): sha(p) for p in sources}, 'current_GT_used': False}
    save(out / 'manifest.json', manifest)
    proposals, final = out / 'proposals.rsdb', out / 'prediction.rsdb'
    records = []
    for name, args in [('pose_proposal', [db, current, proposals, '-v']),
                       ('segment_transfer', [proposals, '-o', final, '-v'])]:
        command = ['wsl', '-d', 'Ubuntu-20.04', '--exec', 'env', 'OMP_NUM_THREADS=1',
                   'timeout', '--signal=TERM', '--kill-after=15', str(cfg['stage_timeout_seconds']),
                   linux(binaries / name)] + [linux(x) if isinstance(x, Path) else x for x in args]
        start = time.perf_counter()
        with (out / (name + '.log')).open('xb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                save(out / 'monitor.json', {'scene': scene, 'mode': mode, 'stage': name,
                    'pid': process.pid, 'alive': True, 'elapsed_s': time.perf_counter()-start,
                    'timeout_s': cfg['stage_timeout_seconds']})
                time.sleep(2)
        record = {'stage': name, 'returncode': process.returncode, 'elapsed_s': time.perf_counter()-start}
        records.append(record)
        print(scene, mode, record, flush=True)
        save(out / 'run.json', {'records': records, 'complete': False})
        if process.returncode != 0:
            save(out / 'monitor.json', {'alive': False, 'failed': record})
            return
    for p, h in manifest['inputs'].items():
        assert sha(p) == h, 'Prior archive unexpectedly changed: ' + p
    latest = [x.split() for x in final.read_text(encoding='utf-8').splitlines() if x.startswith('scene ')][-1]
    prediction = win(latest[3])
    pred = read_ply(prediction)
    assert len(pred) == len(cloud)
    assert all(np.array_equal(pred[f], cloud[f]) for f in ['x', 'y', 'z'])
    classes, counts = np.unique(pred['instance_idx'], return_counts=True)
    summary = []
    for i, count in zip(classes, counts):
        mask = pred['instance_idx'] == i
        cls, n = np.unique(pred['class_idx'][mask], return_counts=True)
        dominant = int(cls[np.argmax(n)])
        if dominant not in cfg['excluded_classes'] and count >= cfg['minimum_points']:
            summary.append({'predicted_id': int(i), 'points': int(count), 'predicted_class': dominant})
    save(out / 'native_groups.json', {'groups': summary, 'group_count': len(summary),
         'coordinate_order_identical': True, 'prediction': str(prediction), 'GT_read': False})
    save(out / 'run.json', {'records': records, 'complete': True, 'current_GT_used': False,
         'scene_count_in_database': len([x for x in final.read_text(encoding='utf-8').splitlines() if x.startswith('scene ')])})
    save(out / 'monitor.json', {'alive': False, 'complete': True})
    save(out / 'prediction_seal.json', {'sha256': {str(p.relative_to(out)): sha(p)
         for p in out.rglob('*') if p.is_file() and p.name != 'prediction_seal.json'}})
    print('SEALED', scene, mode, 'groups', len(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, choices=['scene_a', 'scene_c'])
    ap.add_argument('--output', default='results/rescan_third_period_v1_completed')
    args = ap.parse_args()
    config = Path('configs/rescan_third_period_v1.json')
    cfg = json.loads(config.read_text(encoding='utf-8'))
    for mode in cfg['modes']:
        run(args.scene, mode, cfg, config, Path(args.output))


if __name__ == '__main__':
    main()
