"""TRAIN-only independent measured candidates; never insert annotated positives."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
import time
from pathlib import Path
import numpy as np
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_dataset_v1 import cast,inverse,changed_mask
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'))
    root=Path(cfg['cache_root']);root.mkdir(parents=True,exist_ok=False);(root/'groups').mkdir();(root/'labels').mkdir()
    data=Path(cfg['dataset']);records=[];totals=np.zeros(3,np.int64);start=time.perf_counter()
    sources={p:sha(p) for p in [__file__,'configs/material_correspondence_distribution_v3.json','inspection_decisions/material_correspondence_distribution_v3.py']}
    names=[r['scene'] for r in json.loads((data/'dataset_index.json').read_text(encoding='utf-8')) if r['split']=='train']
    for name in names:
        dest=data/name;meta=json.loads((dest/'public_metadata.json').read_text(encoding='utf-8'));assert meta['split']=='train'
        truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'))
        # The renderer's cast consumes arrays, not model input.
        scene={**truth,'geometry':[np.array(x) for x in truth['geometry']], 'poses':np.array(truth['poses']), 'occluders':np.array(truth['occluders'])}
        all_private={}
        for variant in cfg['variants']:
            groups=[];views=[]
            for epoch in range(3):
                frames,indices=load_scene_epoch(data,name,epoch,variant,shuffle_seed=cfg['seed']+epoch)
                gg=observed_groups(frames,cfg);views.append(frames)
                for slot,g in enumerate(gg):
                    assert len(g['xyz'])>0
                    ids=[];q=[];faces=[]
                    for fi,uv in zip(g['frame'],g['uv']):
                        raw=indices[int(fi)]
                        if raw not in all_private:
                            with np.load(dest/'supervision'/f'frame_{raw:03d}.npz') as f:all_private[raw]={k:f[k] for k in ['instance','material_xyz','face']}
                        gt=all_private[raw];x,y=np.rint(uv).astype(int)
                        ids.append(gt['instance'][y,x]);q.append(gt['material_xyz'][y,x]);faces.append(gt['face'][y,x])
                    g['identity']=np.array(ids);g['q']=np.array(q);g['face']=np.array(faces)
                    assert (g['identity']>=0).all()
                    file=f'{name}_{variant}_e{epoch}_s{slot:02d}.npz'
                    np.savez_compressed(root/'groups'/file,patch=g['patch'].astype(np.float16),area=g['area'],sigma=g['sigma'].astype(np.float32))
                    g['file']='groups/'+file
                groups.append(gg)
            for ea,eb in [(0,1),(0,2),(1,2)]:
                for slot,a in enumerate(groups[ea]):
                    aid=int(np.bincount(a['identity']).argmax())
                    true_slot=int(np.array(truth['permutations'])[eb,aid])
                    # Observe visibility of the same physical material point in
                    # target views, independently of which keypoints were sampled.
                    world=np.zeros_like(a['q'],dtype=float)
                    for identity in np.unique(a['identity']):
                        ix=a['identity']==identity;T=scene['poses'][eb,identity];world[ix]=a['q'][ix]@T[:3,:3].T+T[:3,3]
                    visible=np.zeros(len(world),bool)
                    for f in views[eb]:
                        pc=inverse(world,f['C']);z=pc[:,2];uv=pc[:,:2]/np.maximum(z[:,None],1e-8)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2]
                        h,w=f['depth'].shape;inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
                        ii=np.flatnonzero(inside)
                        if len(ii):
                            ray=cast(scene,eb,f['C'],uv[ii],f['K'])
                            visible[ii]|=(ray['instance']==a['identity'][ii])&(abs(ray['depth_exact']-z[ii])<2e-5)
                    # One true candidate and one observation-selected shape-near
                    # negative. No point is appended/replaced based on truth.
                    extent=np.ptp(a['xyz'],axis=0)
                    errors=[np.linalg.norm(np.sort(np.ptp(b['xyz'],axis=0))-np.sort(extent)) for b in groups[eb]]
                    errors[true_slot]=np.inf;negative=int(np.argmin(errors))
                    for target_slot in [true_slot,negative]:
                        b=groups[eb][target_slot]
                        same=a['identity'][:,None]==b['identity'][None]
                        d2=np.sum((a['q'][:,None]-b['q'][None])**2,axis=2)
                        allowed=same&(d2<=cfg['coverage_radius_m']**2)
                        eligible=visible & (target_slot==true_slot)
                        covered=eligible&allowed.any(1)
                        state=np.where(covered,2,np.where(eligible,1,0)).astype(np.int64)
                        sigma2=np.maximum(cfg['sigma_floor_m']**2,a['sigma'][:,None]**2+b['sigma'][None]**2)
                        teacher=np.exp(-.5*d2/sigma2)*b['area'][None]*allowed*covered[:,None]
                        teacher/=np.maximum(teacher.sum(1,keepdims=True),1e-20)
                        changed=(variant=='changed') & ((ea==2)!=(eb==2)) & np.isin(a['identity'],truth['changed_assets']) & changed_mask(a['q'],a['face'])
                        f=f'{name}_{variant}_{ea}{eb}_{slot:02d}_{target_slot:02d}.npz'
                        np.savez_compressed(root/'labels'/f,states=state,teacher=teacher.astype(np.float16),changed=changed)
                        records.append(dict(source=a['file'],target=b['file'],labels='labels/'+f,scene=name,variant=variant,
                            true_candidate=target_slot==true_slot,states=np.bincount(state,minlength=3).tolist(),
                            repainted_covered=int((changed&covered).sum())))
                        totals+=np.bincount(state,minlength=3)
        save(root/'progress.json',dict(scene=name,episodes=len(records),states=totals,elapsed_s=time.perf_counter()-start))
        print('DEPLOYMENT CACHE',name,len(records),'states',totals.tolist(),round(time.perf_counter()-start,1),flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'index.json',records);save(root/'seal.json',dict(complete=True,train_only=True,positive_injection=False,sources=sources,
        states=totals,episodes=len(records),elapsed_s=time.perf_counter()-start,
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('DEPLOYMENT CACHE SEALED',flush=True)


if __name__=='__main__':main()
