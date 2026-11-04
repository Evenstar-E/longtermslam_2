"""Run all truth-blind predictions, freeze them, THEN allow a separate scorer."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_matching_v1 import extract_epoch, geometric_pools, compare


SOURCES = ['configs/textured_material_comparison_v1.json',
           'inspection_decisions/textured_material_matching_v1.py',
           'inspection_decisions/textured_material_data_io_v1.py',
           'scripts/run_textured_material_comparison_v1.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    def default(a):
        if isinstance(a, np.ndarray): return a.tolist()
        if isinstance(a, np.generic): return a.item()
        raise TypeError(type(a))
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, default=default)+'\n', encoding='utf-8')


def job(scene_name, cfg, root):
    dest = Path(root)/scene_name
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    def progress(message):
        rec = dict(scene=scene_name, message=message, elapsed_s=time.perf_counter()-start)
        save(dest/'progress.json', rec)
        print(scene_name, message, round(rec['elapsed_s'], 1), flush=True)
    geometric = None
    for variant in cfg['variants']:
        frames = [load_scene_epoch(cfg['dataset'], scene_name, e, variant, shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        epochs = [extract_epoch(f, cfg, cfg['seed']+e) for e, f in enumerate(frames)]
        progress(variant+' features ready')
        if geometric is None:
            geometric = geometric_pools(epochs, cfg)
            progress('common depth ICP pools ready')
        predictions, diagnostics = compare(epochs, frames, cfg, geometric, cfg['seed'],
                                           lambda msg: progress(variant+' '+msg))
        for method, pred in predictions.items():
            np.savez_compressed(dest/f'{variant}_{method}.npz', **pred)
        save(dest/f'{variant}_diagnostics.json', diagnostics)
        progress(variant+' all methods saved')
    save(dest/'complete.json', dict(complete=True, elapsed_s=time.perf_counter()-start,
        predictions={p.name:sha(p) for p in sorted(dest.glob('*.npz'))}))
    progress('SCENE COMPLETE')
    return scene_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/textured_material_comparison_v1.json')
    ap.add_argument('--output')
    ap.add_argument('--scenes', nargs='*')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    root = Path(args.output or cfg['output']); root.mkdir(parents=True, exist_ok=False)
    scenes = [x['scene'] for x in json.loads((Path(cfg['dataset'])/'dataset_index.json').read_text(encoding='utf-8'))]
    if args.scenes:
        scenes = args.scenes
    hashes = {p:sha(p) for p in SOURCES}
    save(root/'frozen_protocol.json', dict(config=cfg, scenes=scenes, sources=hashes,
         dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),
         prediction_boundary='No supervision loaded by this program or inference module',
         elapsed_s_not_equal_compute_budget=True))
    start = time.perf_counter()
    if args.workers == 1:
        for s in scenes: job(s,cfg,str(root))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(job,s,cfg,str(root)) for s in scenes]
            for future in as_completed(futures):
                future.result()
    if any(sha(p) != h for p,h in hashes.items()):
        raise RuntimeError('Inference source changed during run')
    save(root/'prediction_seal.json', dict(complete=True, scenes=scenes, elapsed_s=time.perf_counter()-start,
         sources=hashes, files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.npz'))},
         dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json')))
    print('PREDICTIONS SEALED',len(scenes),'scenes',flush=True)


if __name__ == '__main__': main()
