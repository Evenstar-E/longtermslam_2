"""Post-prediction conditional patch diagnostic, NOT full-scene retrieval."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor,metric_patches
from inspection_decisions.material_joint_descriptor_v2 import FactoredMaterialDescriptor
from inspection_decisions.textured_material_data_io_v1 import load_observation
from scripts.build_material_descriptor_train_cache_v1 import select_queries
from scripts.diagnose_material_descriptor_training_v1 import auc
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(2)
    roots=[Path('results/material_joint_descriptor_comparison_v1'),Path('results/material_joint_descriptor_comparison_v2')]
    assert all(json.loads((r/'prediction_seal.json').read_text(encoding='utf-8'))['complete'] for r in roots)
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'));models={}
    for root,files in [('results/material_joint_descriptor_training_v1',['untrained.pt','contrastive_31.pt','full_31.pt']),
                       ('results/material_joint_descriptor_training_v2',['factorized_31.pt','balanced_31.pt','balanced_47.pt','balanced_73.pt'])]:
        for f in files:
            ck=torch.load(Path(root)/f,map_location='cpu',weights_only=True)
            model=(FactoredMaterialDescriptor(64,ck['persistence_prior'],ck['balanced']) if 'persistence_prior' in ck else MaterialDescriptor())
            model.load_state_dict(ck['state_dict']);models[f.replace('.pt','')]=model.eval()
    stats={k:dict(visible=[],m=[],changed=[],c=[],exact=0,physical=0,total=0) for k in models}
    data=Path(cfg['dataset']);index=json.loads((data/'dataset_index.json').read_text(encoding='utf-8'))
    with torch.inference_mode():
        for scene in [r['scene'] for r in index if r['split']!='train']:
            dest=data/scene;pairs=json.loads((dest/'supervision/pairs_v3.json').read_text(encoding='utf-8'))
            for variant in cfg['variants']:
                frames={}
                for pi,p in enumerate(pairs):
                    rng=np.random.default_rng(cfg['seed']+pi)
                    for fi in [p['source'],p['target']]:
                        if fi not in frames:
                            obs=load_observation(dest/'observations'/f'frame_{fi:03d}.npz',variant)
                            with np.load(dest/'supervision'/f'frame_{fi:03d}.npz') as f:gt={k:f[k] for k in ['instance','material_xyz']}
                            frames[fi]=(obs,gt)
                    source,_=frames[p['source']];target,tgt=frames[p['target']]
                    with np.load(dest/p['labels']) as f:labels=dict(f)
                    ix=select_queries(labels,cfg['train_queries'],rng,variant);visible=labels['status'][ix]==0;ii=np.flatnonzero(visible)
                    yy,xx=np.nonzero((target['anonymous_slot']>=0)&target['depth_valid']);rr=rng.choice(len(xx),min(len(xx),cfg['train_random_targets']),replace=False);randomuv=np.c_[xx[rr],yy[rr]]
                    tuv=np.r_[labels['target_uv'][ix[visible]],randomuv]
                    sp,_=metric_patches(source,labels['source_uv'][ix],cfg);tp,_=metric_patches(target,tuv,cfg)
                    identity=np.r_[labels['instance'][ix[visible]],tgt['instance'][randomuv[:,1],randomuv[:,0]]]
                    q=np.r_[labels['material_xyz'][ix[visible]],tgt['material_xyz'][randomuv[:,1],randomuv[:,0]]]
                    near=(labels['instance'][ix,None]==identity[None])&(np.linalg.norm(labels['material_xyz'][ix,None]-q[None],axis=2)<cfg['physical_tolerance_m'])
                    changed=labels['appearance_changed_in_changed_variant'][ix]&(variant=='changed')
                    for name,model in models.items():
                        a,b=model(torch.from_numpy(sp)),model(torch.from_numpy(tp));arg=(a@b.T).argmax(1).numpy();s=stats[name]
                        s['visible'].extend(visible);s['m'].extend(model.context_logits(a,b).sigmoid().numpy())
                        s['exact']+=int((arg[ii]==np.arange(len(ii))).sum());s['physical']+=int(near[ii,arg[ii]].sum());s['total']+=len(ii)
                        s['changed'].extend(changed[ii]);s['c'].extend(model.persistence_logits(a[ii],b[np.arange(len(ii))]).sigmoid().numpy())
            print('PATCH DIAGNOSTIC',scene,flush=True)
    results=[]
    for name,s in stats.items():
        vis=np.array(s['visible']);m=np.array(s['m']);change=np.array(s['changed']);c=np.array(s['c'])
        results.append(dict(model=name,conditional_exact_retrieval=s['exact']/s['total'],conditional_physical_retrieval=s['physical']/s['total'],
            visible_queries=s['total'],matchability_auc=auc(vis,m),matchability_brier=float(np.mean((vis-m)**2)),
            persistence_auc=auc(~change,c),repaint_total=int(change.sum()),repaint_detected_at_05=int((c[change]<.5).sum()),
            unchanged_false_repaint_at_05=int((c[~change]<.5).sum())))
    save(roots[1]/'heldout_patch_diagnostic.json',dict(oracle_candidate_pool=True,not_end_to_end=True,
        description='Targets include visible GT positive and random distractors; physical neighbors count as correct; model inputs contain no GT fields',
        source_sha=sha(__file__),results=results))
    print('PATCH DIAGNOSTIC COMPLETE',results,flush=True)


if __name__=='__main__':main()
