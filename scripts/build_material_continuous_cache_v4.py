"""TRAIN measured surface labels; original anchor sets are never augmented by GT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,observed_groups
from inspection_decisions.material_continuous_correspondence_v4 import surface_grids
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text());cfg=json.loads(Path(v4['base_config']).read_text())
    root=Path(v4['cache_root']);root.mkdir(parents=True,exist_ok=False);(root/'labels').mkdir()
    sources={p:sha(p) for p in [__file__,'configs/material_continuous_correspondence_v4.json','inspection_decisions/material_continuous_correspondence_v4.py']}
    ck=torch.load(Path(cfg['training_root'])/'soft3_31.pt',weights_only=True,map_location='cpu');model=CorrespondenceModel();model.load_state_dict(ck['state_dict']);model.eval()
    data=Path(cfg['dataset']);index=[];start=time.perf_counter();totals=dict(pairs=0,positive=0,original_uncovered_positive=0)
    totals['original_uncovered']=0
    scenes=[r['scene'] for r in json.loads((data/'dataset_index.json').read_text()) if r['split']=='train']
    for name in scenes:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text());private={}
        for variant in cfg['variants']:
            groups=[];grids=[];encoded=[];annotation=[]
            for e in range(3):
                frames,raw=load_scene_epoch(data,name,e,variant,shuffle_seed=cfg['seed']+e);gg=observed_groups(frames,cfg);ss=surface_grids(frames,gg,cfg,v4);ee=[];ann=[]
                for slot,g in enumerate(gg):
                    with torch.inference_mode():ee.append(model.encode(torch.from_numpy(g['patch'])).numpy())
                    identity=[];material=[];surface_q=[];surface_id=[]
                    for k,(fi,uv) in enumerate(zip(g['frame'],g['uv'])):
                        rid=raw[int(fi)]
                        if rid not in private:
                            with np.load(dest/'supervision'/f'frame_{rid:03d}.npz') as f:private[rid]={key:f[key] for key in ['instance','material_xyz']}
                        a=private[rid];x,y=np.rint(uv).astype(int);identity.append(a['instance'][y,x]);material.append(a['material_xyz'][y,x]);x,y=ss[slot]['uv'][k].T
                        surface_q.append(a['material_xyz'][y,x]);surface_id.append(a['instance'][y,x])
                    ann.append(dict(identity=np.array(identity),q=np.array(material),surface_q=np.array(surface_q),surface_id=np.array(surface_id)))
                groups.append(gg);grids.append(ss);encoded.append(ee);annotation.append(ann)
            for ea,eb in [(0,1),(0,2),(1,2)]:
                for i,a in enumerate(groups[ea]):
                    aid=int(np.bincount(annotation[ea][i]['identity']).argmax());j=truth['permutations'][eb][aid]
                    # True object and observed shape-near negative, never a GT
                    # chosen anchor or inserted target point.
                    extent=np.sort(np.ptp(a['xyz'],axis=0));dist=[np.linalg.norm(np.sort(np.ptp(b['xyz'],axis=0))-extent) for b in groups[eb]];dist[j]=np.inf
                    for k in [j,int(np.argmin(dist))]:
                        fn=f'{name}_{variant}_{ea}{eb}_{i:02d}_{k:02d}.npz'
                        with np.load(Path(cfg['cache_root'])/'labels'/fn) as old_labels:old_states=old_labels['states']
                        nearest=np.argsort(-(encoded[ea][i]@encoded[eb][k].T),axis=1)[:,:v4['coarse_candidates']]
                        aq=annotation[ea][i];bq=annotation[eb][k];q=bq['surface_q'][nearest];ids=bq['surface_id'][nearest]
                        d2=((aq['q'][:,None,None]-q)**2).sum(3);valid=grids[eb][k]['valid'][nearest]
                        allowed=valid&(old_states[:,None,None]>0)&(ids==aq['identity'][:,None,None])&(d2<=v4['teacher_radius_m']**2)
                        teacher=np.exp(-.5*d2/v4['teacher_sigma_m']**2)*allowed;mass=teacher.sum(2);positive=mass>1e-8
                        teacher/=np.maximum(mass[:,:,None],1e-20)
                        totals['original_uncovered']+=int((old_states==1).sum())
                        totals['original_uncovered_positive']+=int(((old_states==1)&positive.any(1)).sum())
                        np.savez_compressed(root/'labels'/fn,anchors=nearest.astype(np.int16),teacher=teacher.astype(np.float16),positive=positive,mask=valid)
                        source=f'groups/{name}_{variant}_e{ea}_s{i:02d}.npz';target=f'groups/{name}_{variant}_e{eb}_s{k:02d}.npz'
                        index.append(dict(scene=name,variant=variant,source=source,target=target,labels='labels/'+fn,positive=int(positive.sum()),pairs=int(positive.size),true_candidate=k==j))
                        totals['pairs']+=positive.size;totals['positive']+=int(positive.sum())
        print('V4 CACHE',name,totals,round(time.perf_counter()-start,1),flush=True)
    # No inference or DEV labels were read. Old cache provides patch tensors.
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'index.json',index);save(root/'seal.json',dict(complete=True,train_only=True,positive_injection=False,sources=sources,
        anchor_cache=str(cfg['cache_root']),anchor_cache_seal_sha=sha(Path(cfg['cache_root'])/'seal.json'),coarse_model_sha=sha(Path(cfg['training_root'])/'soft3_31.pt'),
        records=len(index),totals=totals,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))


if __name__=='__main__':main()
