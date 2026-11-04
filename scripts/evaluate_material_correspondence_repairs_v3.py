"""Actual sampled-set diagnostics and TRAIN-calibrated repaint validation."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,observed_groups
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_dataset_v1 import inverse,cast,changed_mask
from scripts.train_material_appearance_v3 import auc,physical_hash
from scripts.run_textured_material_comparison_v1 import save,sha


def annotate(frames,indices,dest,cfg,private):
    groups=observed_groups(frames,cfg)
    for g in groups:
        ids=[];q=[];faces=[]
        for fi,uv in zip(g['frame'],g['uv']):
            raw=indices[int(fi)]
            if raw not in private:
                with np.load(dest/'supervision'/f'frame_{raw:03d}.npz') as f:private[raw]={k:f[k] for k in ['instance','material_xyz','face']}
            x,y=np.rint(uv).astype(int);f=private[raw];ids.append(f['instance'][y,x]);q.append(f['material_xyz'][y,x]);faces.append(f['face'][y,x])
        g.update(identity=np.array(ids),q=np.array(q),face=np.array(faces))
    return groups


def visibility(a,scene,epoch,frames):
    world=np.zeros_like(a['q'],float)
    for identity in np.unique(a['identity']):
        ix=a['identity']==identity;T=scene['poses'][epoch,identity];world[ix]=a['q'][ix]@T[:3,:3].T+T[:3,3]
    visible=np.zeros(len(world),bool)
    for f in frames:
        pc=inverse(world,f['C']);z=pc[:,2];uv=pc[:,:2]/np.maximum(z[:,None],1e-8)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2]
        h,w=f['depth'].shape;ii=np.flatnonzero((z>0)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1))
        if len(ii):
            ray=cast(scene,epoch,f['C'],uv[ii],f['K']);visible[ii]|=(ray['instance']==a['identity'][ii])&(abs(ray['depth_exact']-z[ii])<2e-5)
    return visible


def main():
    torch.set_num_threads(1);cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'))
    root=Path(cfg['comparison_root']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
    models={};appearance={};thresholds={}
    for mode,seed in [('hard2',31),('soft2',31),('soft3',31),('soft3',47),('soft3',73)]:
        ck=torch.load(Path(cfg['training_root'])/f'{mode}_{seed}.pt',weights_only=True,map_location='cpu');model=CorrespondenceModel();model.load_state_dict(ck['state_dict']);models[f'{mode}_{seed}']=model.eval()
    for seed in cfg['seeds']:
        ck=torch.load(Path('results/material_appearance_training_v3')/f'soft3_{seed}.pt',weights_only=True,map_location='cpu');model=CorrespondenceModel();model.load_state_dict(ck['state_dict']);model.eval()
        assert physical_hash(model)==physical_hash(models[f'soft3_{seed}'])
        appearance[f'repaired_{seed}']=model;thresholds[f'repaired_{seed}']=ck['appearance_threshold']
    appearance['before_31']=models['soft3_31']
    # Matched operating-point control: baseline also receives a TRAIN-only 5%FPR threshold.
    cache=Path(cfg['cache_root']);index=json.loads((cache/'index.json').read_text(encoding='utf-8'))
    @lru_cache(maxsize=128)
    def group(file):
        with np.load(cache/file) as f:return f['patch']
    train_scores=[];train_y=[];model=appearance['before_31']
    with torch.inference_mode():
        for r in index:
            if not r['true_candidate']:continue
            with np.load(cache/r['labels']) as f:ii=np.flatnonzero(f['states']==2);jj=f['teacher'][ii].argmax(1);y=f['changed'][ii]
            if not len(ii):continue
            a=model.encode_appearance(torch.from_numpy(group(r['source'])[ii]).float());b=model.encode_appearance(torch.from_numpy(group(r['target'])[jj]).float())
            train_scores.extend(model.change_logits(a,b).sigmoid().numpy());train_y.extend(y)
    train_y=np.array(train_y);train_scores=np.array(train_scores)
    thresholds['before_31']=float(np.quantile(train_scores[~train_y],.95))
    save(root/'appearance_operating_points.json',dict(train_only=True,target_unchanged_fpr=.05,thresholds=thresholds,source_sha=sha(__file__)))
    state_stats={k:dict(confusion=np.zeros((3,3),int),visible=0,covered=0,top1=0,top3=0,brier_sum=0.,visible_count=0) for k in models}
    app_stats={k:dict(y=[],scores=[]) for k in appearance};support=np.zeros(3,int)
    data=Path(cfg['dataset']);names=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'))['scenes']
    with torch.inference_mode():
        for name in names:
            dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'))
            scene={**truth,'geometry':[np.array(x) for x in truth['geometry']],'poses':np.array(truth['poses']),'occluders':np.array(truth['occluders'])};private={}
            for variant in cfg['variants']:
                frames=[];groups=[]
                for e in range(3):
                    f,ii=load_scene_epoch(data,name,e,variant,shuffle_seed=cfg['seed']+e);frames.append(f);groups.append(annotate(f,ii,dest,cfg,private))
                enc={k:[[model.encode(torch.from_numpy(g['patch'])) for g in gg] for gg in groups] for k,model in models.items()}
                for ea,eb in [(0,1),(0,2),(1,2)]:
                    for i,a in enumerate(groups[ea]):
                        identity=int(np.bincount(a['identity']).argmax());j=truth['permutations'][eb][identity];b=groups[eb][j]
                        vis=visibility(a,scene,eb,frames[eb]);d2=((a['q'][:,None]-b['q'][None])**2).sum(2)
                        allowed=(a['identity'][:,None]==b['identity'][None])&(d2<=cfg['coverage_radius_m']**2)
                        covered=vis&allowed.any(1);states=np.where(covered,2,np.where(vis,1,0));support+=np.bincount(states,minlength=3)
                        sigma2=np.maximum(cfg['sigma_floor_m']**2,a['sigma'][:,None]**2+b['sigma'][None]**2)
                        teacher=np.exp(-.5*d2/sigma2)*b['area'][None]*allowed;target=teacher.argmax(1)
                        change=(variant=='changed')&((ea==2)!=(eb==2))&np.isin(a['identity'],truth['changed_assets'])&changed_mask(a['q'],a['face'])
                        for key,model in models.items():
                            logq,sl=model.distribution(enc[key][ea][i],enc[key][eb][j],torch.from_numpy(b['area']).float());p=sl.softmax(1).numpy();order=np.argsort(-logq.numpy(),axis=1)[:,:3];s=state_stats[key]
                            s['visible']+=int(vis.sum());s['covered']+=int(covered.sum());ii=np.flatnonzero(covered)
                            s['top1']+=int(allowed[ii,order[ii,0]].sum());s['top3']+=int(allowed[ii[:,None],order[ii]].any(1).sum())
                            cov=p[:,2]/np.maximum(p[:,1]+p[:,2],1e-8) if key.startswith('soft3') else p[:,2]
                            s['brier_sum']+=float(((cov[vis]-covered[vis])**2).sum());s['visible_count']+=int(vis.sum())
                            for yy,pp in zip(states,p.argmax(1)):s['confusion'][yy,pp]+=1
                        ii=np.flatnonzero(covered)
                        if len(ii):
                            for key,model in appearance.items():
                                aa=model.encode_appearance(torch.from_numpy(a['patch'][ii]));bb=model.encode_appearance(torch.from_numpy(b['patch'][target[ii]]))
                                app_stats[key]['y'].extend(change[ii]);app_stats[key]['scores'].extend(model.change_logits(aa,bb).sigmoid().numpy())
            print('ACTUAL SAMPLING DIAGNOSTIC',name,flush=True)
    point=[];app=[]
    for key,s in state_stats.items():
        point.append(dict(model=key,visible=s['visible'],covered=s['covered'],actual_sample_top1=s['top1']/s['covered'],actual_sample_top3=s['top3']/s['covered'],
            coverage_brier_given_visible=s['brier_sum']/s['visible_count'],confusion=s['confusion']))
    for key,s in app_stats.items():
        y=np.array(s['y']);p=np.array(s['scores']);pred=p>thresholds[key]
        app.append(dict(model=key,threshold=thresholds[key],auc=auc(y,p),changed_total=int(y.sum()),unchanged_total=int((~y).sum()),
            repaint_recall=float(pred[y].mean()),unchanged_fpr=float(pred[~y].mean()),repaint_detected=int(pred[y].sum()),unchanged_false_positive=int(pred[~y].sum())))
    save(root/'repair_diagnostics.json',dict(source_sha=sha(__file__),actual_independent_candidates=True,GT_identity_conditioned=True,positive_injection=False,
        states=support,point_metrics=point,appearance_metrics=app,appearance_isolated_physical_weights=True))
    print('REPAIR DIAGNOSTICS COMPLETE',point,app,flush=True)


if __name__=='__main__':main()
