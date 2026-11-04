"""Only AFTER prediction seal: full material displacement and continuous links."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv
from pathlib import Path
from functools import lru_cache
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14 import configuration,METHODS,setup,load,dense_records,DensePair,ContinuousTriple,PAIRS
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from inspection_decisions.textured_material_matching_v1 import apply,world_points
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));rows=[];oracle=[];links=[];mechanism=[];maxerror=0.;paths=0;centers=0
    for v in cfg['variants']:
        _,frames,ids,bc,_,_,views,banks,_=setup(cfg,v);dense=dense_records(views);cache=load(root/f'{v}_cache.npz');graph=load(root/f'{v}_search_graph.npz');payload=load(root/f'{v}_proposals.npz');fits=load(root/f'{v}_fits.npz');old=load(Path(cfg['retained_root'])/f'{v}_cache.npz');og=load(Path(cfg['retained_root'])/f'{v}_search_graph.npz');oldgen=load(Path(cfg['retained_root'])/f'{v}_proposals.npz');prev=load(Path(cfg['previous_root'])/f'{v}_cache.npz');plook={tuple(t):h for h,t in enumerate(prev['triples'])}
        assert np.array_equal(cache['triples'],old['triples']) and np.array_equal(graph['states'],og['states']);checks+=1;lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
        for key,path in fits.items():
            if key.endswith('_path'):assert np.max(np.diff(path),initial=0)<=1e-12;checks+=1;paths+=1
        for e,row in enumerate(banks):
            for slot,a in enumerate(row):
                assert len(a['xyz'])==512;checks+=1;centers+=len(a['xyz'])
                for fi in np.unique(a['frame']):
                    h=a['frame']==fi;f=frames[e][int(fi)];uv=a['uv'][h];assert np.max(abs(world_points(f,uv)-a['xyz'][h]))<1e-12 and np.array_equal(f['rgb'][uv[:,1],uv[:,0]]/255.,a['rgb'][h]);checks+=1
        for state,value in zip(graph['states'],graph['values']):
            assert all(np.array_equal(np.sort(state[:,e]),np.arange(24)) for e in range(2));val=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(8));assert np.max(abs(val-value))<1e-8;checks+=2
        for h,triple in enumerate(cache['triples']):
            U=payload[f'{h}_transforms'];R=U[:,:,:3,:3];assert np.isfinite(U).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
            retained=list(oldgen[f'{h}_transforms'])+list(old['transforms'][h]);key=tuple(triple)
            if key in plook:retained+=list(prev['transforms'][plook[key]])
            for T in retained:assert any(np.max(abs(T-Q))<1e-9 for Q in U);checks+=1
            assert np.max(abs(payload[f'{h}_values'].min(0)-cache['costs'][h,1:]))<1e-8;checks+=1
        @lru_cache(maxsize=128)
        def pair(pe,i,j):
            e,f=PAIRS[pe];return DensePair(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],cfg)
        def scorer(i,j,k):return ContinuousTriple([pair(0,i,j),pair(1,i,k),pair(2,j,k)],cfg)
        private_chart_material(scene,banks,ids);frame_index=[{id(f):h for h,f in enumerate(row)} for row in frames]
        @lru_cache(maxsize=16)
        def label(e,fi):
            rec=records[int(ids[e][fi])]
            with np.load(scene/rec['labels']) as f:return f['material_xyz'],f['instance']
        stageby={}
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);key=i,j,k;h=lookup.get(key);data={}
            if h is not None:
                U=payload[f'{h}_transforms'];stage=payload[f'{h}_stage'];x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);error=np.array([physical_error(*T,x,y1,y2) for T in U])
                for name,limit in [('retained',0),('global',1),('continuous',2)]:data[name]=float(error[stage<=limit].min())
                best=int(error.argmin());GT=np.array([poses[e,i]@inverse(poses[0,i]) for e in [1,2]]);true=scorer(i,j,k).score(GT[None])[0];bestvalue=payload[f'{h}_values'][best]
                mechanism.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),best_error_m=float(error[best]),best_stage=int(stage[best]),true_continuous_pair=float(true[3]),true_shared=float(true[6]),best_continuous_pair=float(bestvalue[3]),best_shared=float(bestvalue[6])))
            else:data={k:None for k in ['retained','global','continuous']}
            stageby[i]=data;oracle.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),correct_identity_searched=int(h is not None),**{name+'_error_m':value for name,value in data.items()}))
        for col,m in enumerate(METHODS):
            p=load(root/f'{v}_{m}.npz');state=int(p['search_state']);assert state==int(graph['best'][col]) and np.array_equal(p['target_slots'],graph['states'][state]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,p,truth,anchors,bc)
            for i,(j,k) in enumerate(p['target_slots']):
                j,k=int(j),int(k);h=lookup[i,j,k];T=p['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);checks+=1
                if col:
                    sc=scorer(i,j,k);value,detail=sc.evaluate(T,True);error=abs(float(value[col-1])-float(p['objectives'][i]));assert error<1e-8;maxerror=max(maxerror,error);checks+=1
                    if col>=3:
                        corecol=col-1;mode=detail['fixed' if corecol==2 else 'shift']
                        for pe,((e,f),pr) in enumerate(zip(PAIRS,sc.pairs)):
                            for direction in [0,1]:
                                source_epoch,target_epoch=(e,f) if direction==0 else (f,e);a=pr.a if direction==0 else pr.b;fs=pr.fb if direction==0 else pr.fa;r=mode[pe][1][direction]
                                if pe<2 and direction==0 and corecol>=4:r=detail['groups'][corecol-4][pe]
                                active=r['valid']&(r['kernel']>1e-6);n=int(active.sum());correct=0
                                for fi,frame in enumerate(fs):
                                    ix=np.flatnonzero(active&(r['frame']==fi))
                                    if not len(ix):continue
                                    uv=np.rint(r['uv'][ix]).astype(int);uv[:,0]=np.clip(uv[:,0],0,frame['depth'].shape[1]-1);uv[:,1]=np.clip(uv[:,1],0,frame['depth'].shape[0]-1);qgrid,idgrid=label(target_epoch,frame_index[target_epoch][id(frame)]);q=qgrid[uv[:,1],uv[:,0]];identity=idgrid[uv[:,1],uv[:,0]];correct+=int(((a['physical_id'][ix]==identity)&(np.linalg.norm(a['q'][ix]-q,axis=1)<=.015)).sum())
                                links.append(dict(variant=v,method=m,asset=i,pair=pe,direction=direction,rear=int(perm[2,i]%3==0),active_links=n,correct_links=correct,material_radius_m=.015,active_threshold=1e-6))
                o=stageby[i]['retained' if not col else 'continuous'];rr[i].update(rear=int(perm[2,i]%3==0),oracle_error_m=o,oracle_qualified=int(o is not None and o<=.05));assert rr[i]['qualified']<=rr[i]['oracle_qualified'];checks+=1
            rows+=rr
        a=load(Path(cfg['retained_root'])/f'{v}_independent_material.npz');b=load(root/f'{v}_retained_v12.npz');assert np.array_equal(a['target_slots'],b['target_slots']) and np.array_equal(a['transforms'],b['transforms']);checks+=1
        i,j,k=map(int,cache['triples'][0]);full=scorer(i,j,k).score(payload['0_transforms']);assert np.max(abs(full-payload['0_values']))<1e-8;checks+=1
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_by_stage.csv',oracle);write_csv(root/'active_material_links.csv',links);write_csv(root/'pose_mechanism.csv',mechanism);summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS:
            rr=[r for r in rows if r['variant']==v and r['method']==m];good=[r for r in rr if r['both_ids']];ll=[r for r in links if r['variant']==v and r['method']==m];n=sum(r['active_links'] for r in ll);c=sum(r['correct_links'] for r in ll);summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=len(good),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_qualified=sum(r['oracle_qualified'] for r in rr),correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in good])) if good else None,active_links=n,correct_links=c,active_link_precision=c/n if n else None))
        for a,b in [('retained_v12','dense_fusion'),('dense_geometry','dense_fusion'),('projection_fixed','continuous_pair'),('continuous_pair','material_independent'),('material_independent','material_track_independent'),('material_track_independent','material_track_shared')]:
            aa={r['asset']:r for r in rows if r['variant']==v and r['method']==a};bb={r['asset']:r for r in rows if r['variant']==v and r['method']==b};common=[i for i in aa if aa[i]['both_ids'] and bb[i]['both_ids']];comparisons.append(dict(variant=v,baseline=a,candidate=b,gained=[i for i in aa if bb[i]['qualified'] and not aa[i]['qualified']],lost=[i for i in aa if aa[i]['qualified'] and not bb[i]['qualified']],common_correct_ids=len(common),paired_worst_error_change_median_m=float(np.median([bb[i]['worst_error_m']-aa[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(primary_rows=len(rows),checks=checks,fit_paths_verified=paths,source_centers_verified=centers,max_selected_objective_error=maxerror,summary=summary,comparisons=comparisons,scope=cfg['contract']));print(json.dumps(dict(checks=checks,summary=summary,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
