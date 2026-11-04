"""Export common three-period native-fragment inputs before reading capture2 truth."""
import json
import shutil
from pathlib import Path
import numpy as np
from scripts.run_rescan_third_period_v1 import sha, save
from scripts.prepare_real_temporal_v1 import read_ply


def run_root(scene,mode,progress=False):
    name='rescan_third_period_progress_v1' if progress and scene=='scene_a' else 'rescan_third_period_v1_completed'
    return Path('results')/name/scene/mode


def verify_runs(modes=None,progress=False):
    base = Path('results/rescan_third_period_v1_completed')
    for scene in ['scene_a', 'scene_c']:
        for mode in (modes or ['initial_archive', 'inherited_unfused', 'inherited_protected']):
            root = run_root(scene,mode,progress)
            assert json.loads((root/'run.json').read_text(encoding='utf-8'))['complete'], root
            for p,h in json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['sha256'].items():
                assert sha(root/p) == h, root/p
            manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            for group in ['sources', 'inputs']:
                for p,h in manifest[group].items():
                    assert sha(p) == h, p
    return base


def main():
    base = verify_runs(['initial_archive'])
    prior = Path('data/livingscenes_native_masks_v1')
    old = json.loads((prior/'public.json').read_text(encoding='utf-8'))
    root = Path('data/rescan_three_period_native_v1')
    root.mkdir(exist_ok=False)
    public, mapping, inputs = {}, {}, []
    for si, scene in enumerate(['scene_a', 'scene_c']):
        public[scene] = {'0':old[scene]['0'], '1':old[scene]['1'], '2':[]}
        for period in ['0','1']:
            for record in public[scene][period]:
                src, dst = prior/record['path'], root/record['path']
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src,dst)
                assert sha(src) == sha(dst)
        mode = base/scene/'initial_archive'
        rec = json.loads((mode/'native_groups.json').read_text(encoding='utf-8'))
        prediction = Path(rec['prediction'])
        points = read_ply(prediction)
        identifiers = np.unique(points['instance_idx'])
        np.random.default_rng(20261024+si).shuffle(identifiers)
        mapping[scene] = {}
        for identifier in identifiers:
            mask = points['instance_idx'] == identifier
            classes,counts = np.unique(points['class_idx'][mask],return_counts=True)
            cls = int(classes[np.argmax(counts)])
            if cls in [0,1,2,22]:
                continue
            token = f'n{len(public[scene]["2"]):03d}'
            dest = root/'public'/scene/'2'/(token+'.npz')
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest, xyz=np.column_stack([points[f][mask] for f in ['x','y','z']]).astype(float),
                rgb=np.column_stack([points[f][mask] for f in ['red','green','blue']]),point_indices=np.flatnonzero(mask))
            public[scene]['2'].append({'token':token,'semantic_class':cls,'raw_points':int(mask.sum()),
                'eligible':bool(mask.sum()>=128),'path':str(dest.relative_to(root))})
            mapping[scene][token] = int(identifier)
        inputs.append({'path':str(prediction),'sha256':sha(prediction)})
    save(root/'public.json',public)
    (root/'evaluation').mkdir()
    save(root/'evaluation/native_token_mapping_capture2.json',mapping)
    save(root/'contract.json',{'sources':inputs,'source_script_sha':sha(__file__),
        'period2_frontend':'initial_archive arm fixed before GT evaluation; not selected by performance',
        'current_GT_masks':False,'current_GT_classes':False,'current_predicted_identity_removed':True,
        'scope':'Three-period native fragments. Frontend still historically conditioned and may mix/split physical assets. No continuous tracking or displacement metric claimed.'})
    save(root/'seal.json',{'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('EXPORTED', {s:{t:sum(r['eligible'] for r in rows) for t,rows in periods.items()} for s,periods in public.items()})


if __name__=='__main__':
    main()
