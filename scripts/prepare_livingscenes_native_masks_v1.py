"""Export anonymous masks produced by native Rescan, without reading current GT."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply

root=Path('data/livingscenes_native_masks_v1');root.mkdir(exist_ok=False)
original=Path('data/real_cross_session_v1/prepared_v2');old=json.loads((original/'public.json').read_text(encoding='utf-8'))
cfg=json.loads(Path('configs/livingscenes_real_v1.json').read_text(encoding='utf-8'));public={};mapping={};inputs=[]
native={'scene_a':Path('results/rescan_author_v1/inference_long_scene_a_v2'),'scene_c':Path('results/rescan_author_v1/inference_memory_port')}
for si,(scene,base) in enumerate(native.items()):
    record=json.loads((base/'run.json').read_text(encoding='utf-8'))
    assert any(r['scene']==scene and r['stage']=='segment_transfer' and r['returncode']==0 for r in record['records'])
    prediction=base/scene/'predictions/prediction.ply';data=read_ply(prediction)
    inputs.append({'path':str(prediction),'sha256':hashlib.sha256(prediction.read_bytes()).hexdigest()})
    public[scene]={'0':old[scene]['0'],'1':[]};mapping[scene]={}
    for h in public[scene]['0']:
        target=root/h['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/h['path'],target)
    identifiers=np.unique(data['instance_idx']);np.random.default_rng(cfg['seed']+si).shuffle(identifiers)
    for identifier in identifiers:
        mask=data['instance_idx']==identifier;classes,counts=np.unique(data['class_idx'][mask],return_counts=True);cls=int(classes[np.argmax(counts)])
        if cls in [0,1,2,22]:continue
        token=f'n{len(public[scene]["1"]):03d}';target=root/'public'/scene/'1'/(token+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        xyz=np.column_stack([data[n][mask] for n in ['x','y','z']]).astype(float);rgb=np.column_stack([data[n][mask] for n in ['red','green','blue']])
        np.savez_compressed(target,xyz=xyz,rgb=rgb,point_indices=np.flatnonzero(mask))
        public[scene]['1'].append({'token':token,'semantic_class':cls,'raw_points':int(mask.sum()),'eligible':bool(mask.sum()>=128),'path':str(target.relative_to(root))})
        mapping[scene][token]=int(identifier)
(root/'public.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
(root/'evaluation').mkdir();(root/'evaluation/native_mask_token_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
(root/'acquisition.json').write_text(json.dumps({'sources':inputs,'current_GT_read':False,'current_masks':'native Rescan predictions, IDs anonymized','historical_points':'initial capture0, not model database updated with current scan'},indent=2),encoding='utf-8')
cfg['public_root']=str(root);cfg['spaces']=['scene_a','scene_c'];cfg['scope']='Same published prior on actual anonymized native predicted masks/classes. Development baseline integration, not a new algorithm. No current GT masks supplied.'
cfg['pose_probes']=None
Path('configs/livingscenes_native_masks_v1.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
runner=Path('scripts/run_livingscenes_real_v1.py').read_text(encoding='utf-8')
runner=runner.replace('results/livingscenes_real_v1','results/livingscenes_native_masks_v1').replace('configs/livingscenes_real_v1.json','configs/livingscenes_native_masks_v1.json').replace("'uses_current_GT_masks':True","'uses_current_GT_masks':False")
Path('scripts/run_livingscenes_native_masks_v1.py').write_text(runner,encoding='utf-8')
print({s:{k:sum(x['eligible'] for x in v) for k,v in d.items()} for s,d in public.items()})
