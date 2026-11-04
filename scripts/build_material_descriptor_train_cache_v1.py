"""Build bounded metric-patch cache from TRAIN ONLY; preserve original dataset."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_observation
from inspection_decisions.material_joint_descriptor_v1 import metric_patches
from scripts.run_textured_material_comparison_v1 import save,sha


def training_scene(path):
    meta=json.loads((path/'public_metadata.json').read_text(encoding='utf-8'))
    if meta['split']!='train':raise ValueError('Cache builder refuses non-training supervision')
    return meta


def select_queries(labels,count,rng,variant):
    status=labels['status'];changed=labels['appearance_changed_in_changed_variant']&(variant=='changed')
    groups=[np.flatnonzero((status==0)&changed),np.flatnonzero(status==1),np.flatnonzero(status==2),np.flatnonzero((status==0)&~changed)]
    selected=[]
    for ids,n in zip(groups,[8,8,8,24]):
        if len(ids):selected.extend(rng.choice(ids,min(n,len(ids)),replace=False).tolist())
    rest=np.setdiff1d(np.arange(len(status)),selected)
    if len(selected)<count:selected.extend(rng.choice(rest,min(count-len(selected),len(rest)),replace=False).tolist())
    return np.array(selected[:count])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/material_joint_descriptor_v1.json');args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(cfg['cache_root']);root.mkdir(parents=True,exist_ok=False)
    data=Path(cfg['dataset']);index=json.loads((data/'dataset_index.json').read_text());records=[];start=time.perf_counter()
    sources={p:sha(p) for p in [args.config,__file__,'inspection_decisions/material_joint_descriptor_v1.py']}
    for scene in [r['scene'] for r in index if r['split']=='train']:
        dest=data/scene;training_scene(dest)
        pairs=json.loads((dest/'supervision/pairs_v3.json').read_text())
        for variant in cfg['variants']:
            cache={}
            for pi,p in enumerate(pairs):
                rng=np.random.default_rng(cfg['seed']+pi)
                for fi in [p['source'],p['target']]:
                    if fi not in cache:
                        obs=load_observation(dest/'observations'/f'frame_{fi:03d}.npz',variant)
                        with np.load(dest/'supervision'/f'frame_{fi:03d}.npz') as f:truth={k:f[k] for k in ['instance','material_xyz']}
                        cache[fi]=(obs,truth)
                source,source_truth=cache[p['source']];target,target_truth=cache[p['target']]
                with np.load(dest/p['labels']) as f:labels=dict(f)
                ii=select_queries(labels,cfg['train_queries'],rng,variant)
                visible=labels['status'][ii]==0;visix=np.flatnonzero(visible)
                source_uv=labels['source_uv'][ii];target_uv=labels['target_uv'][ii[visible]]
                yy,xx=np.nonzero((target['anonymous_slot']>=0)&target['depth_valid'])
                choose=rng.choice(len(xx),min(len(xx),cfg['train_random_targets']),replace=False)
                random_uv=np.c_[xx[choose],yy[choose]]
                all_uv=np.concatenate([target_uv,random_uv])
                sp,_=metric_patches(source,source_uv,cfg);tp,_=metric_patches(target,all_uv,cfg)
                ids=np.r_[labels['instance'][ii[visible]],target_truth['instance'][random_uv[:,1],random_uv[:,0]]]
                q=np.r_[labels['material_xyz'][ii[visible]],target_truth['material_xyz'][random_uv[:,1],random_uv[:,0]]]
                near=(labels['instance'][ii,None]==ids[None])&(np.linalg.norm(labels['material_xyz'][ii,None]-q[None],axis=2)<cfg['physical_tolerance_m'])
                positive=np.zeros(len(ii),np.int64);positive[visix]=np.arange(len(visix))
                changed=labels['appearance_changed_in_changed_variant'][ii]&(variant=='changed')
                name=f'{scene}_{variant}_{pi:03d}.npz'
                np.savez_compressed(root/name,source=sp.astype(np.float16),target=tp.astype(np.float16),
                    visible=visible,positive=positive,changed=changed,near=near)
                records.append(dict(file=name,scene=scene,variant=variant,queries=len(ii),visible=int(visible.sum()),
                    repaint_visible=int((visible&changed).sum()),outside=int((labels['status'][ii]==1).sum()),occluded=int((labels['status'][ii]==2).sum())))
        save(root/'progress.json',dict(completed_scene=scene,cases=len(records),elapsed_s=time.perf_counter()-start))
        print('CACHE',scene,'cases',len(records),round(time.perf_counter()-start,1),flush=True)
    assert len({r['scene'] for r in records})==cfg['train_scenes']
    save(root/'index.json',records)
    save(root/'seal.json',dict(complete=True,split='train',sources=sources,elapsed_s=time.perf_counter()-start,
        totals={k:sum(r[k] for r in records) for k in ['queries','visible','repaint_visible','outside','occluded']},
        files={p.name:sha(p) for p in root.glob('*.npz')}))
    print('CACHE SEALED',len(records),flush=True)


if __name__=='__main__':main()
