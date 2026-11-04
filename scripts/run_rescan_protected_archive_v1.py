"""Pre-evaluation intervention: inherited layouts but immutable initial asset surfaces.

An ordinary memory-update ablation, not a new probabilistic association method.
No current labels or evaluation statistics are read.
"""
import argparse
import json
import shutil
from pathlib import Path
from scripts.run_rescan_third_period_v1 import run, save, win, linux, sha, database_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, choices=['scene_a', 'scene_c'])
    args = ap.parse_args()
    cfg = json.loads(Path('configs/rescan_third_period_v1.json').read_text(encoding='utf-8'))
    prior = Path(cfg['prior_roots'][args.scene])
    initial = (prior / 'history.rsdb').read_text(encoding='utf-8').splitlines()
    inherited = (prior / 'prediction.rsdb').read_text(encoding='utf-8').splitlines()
    source_folder = win(next(x.split(maxsplit=1)[1] for x in initial if x.startswith('model_folder ')))
    inherited_folder = win(next(x.split(maxsplit=1)[1] for x in inherited if x.startswith('model_folder ')))
    original_objects = {int(t[2]): source_folder/t[1] for t in map(str.split, initial) if t and t[0]=='object'}
    root = Path('data/rescan_protected_archive_v1') / args.scene
    root.mkdir(parents=True, exist_ok=False)
    models = root / 'models'
    models.mkdir()
    records = []
    for t in map(str.split, inherited):
        if not t or t[0] != 'object':
            continue
        uid = int(t[2])
        source = original_objects.get(uid, inherited_folder/t[1])
        dest = models / t[1]
        shutil.copyfile(source, dest)
        assert sha(source) == sha(dest)
        records.append({'uid': uid, 'source': str(source), 'destination': str(dest),
                        'sha256': sha(source), 'initial_surface_restored': uid in original_objects})
    modified = [f'model_folder {linux(models)}' if x.startswith('model_folder ') else x for x in inherited]
    (root / 'prediction.rsdb').write_text('\n'.join(modified)+'\n', encoding='utf-8')
    assert [x for x in modified if not x.startswith('model_folder ')] == [x for x in inherited if not x.startswith('model_folder ')]
    save(root / 'intervention.json', {'scope': 'Initial object surfaces restored in inherited database; scene records, object ordering, poses, classes and new assets retained. Known ordinary immutable-memory ablation, not novel.',
        'models': records, 'source_sha': sha(__file__), 'old_databases': {
            str(prior/'history.rsdb'): sha(prior/'history.rsdb'),
            str(prior/'prediction.rsdb'): sha(prior/'prediction.rsdb')},
        'current_GT_read': False})
    cfg['prior_roots'][args.scene] = str(root)
    cfg['modes'] = ['inherited_protected']
    cfg['scope'] = 'Inherited-unfused layout, initial model shapes restored, new models retained. Ordinary surface-update intervention; no full Poisson-fused reproduction or novelty claim.'
    config = root / 'configuration.json'
    save(config, cfg)
    run(args.scene, 'inherited_protected', cfg, config, Path('results/rescan_third_period_v1_completed'))


if __name__ == '__main__':
    main()
