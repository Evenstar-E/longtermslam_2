"""Post-seal task evaluation and independent selected-energy reconstruction."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_matching_v1 import apply
from inspection_decisions.material_partial_track_transport_v7 import PairTransport,joint_values,selected_edges,PAIRS,inverse
from scripts.run_material_partial_track_transport_v7 import METHODS,setup
from scripts.run_material_observation_registration_v7 import make_scorer
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    cfg=json.loads(Path('configs/material_partial_track_transport_v7.json').read_text(encoding='utf-8'));root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));data=Path(ic['dataset']);release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    scene=data/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(scene);poses=np.array(truth['poses']);perm=np.array(truth['permutations']);rows=[];diagnostics=[];max_error=0.;fitcount=0;improvedfits=0
    for v in cfg['variants']:
        _,bc,ep,groups,views,banks,edges=setup(cfg,v)
        with np.load(root/f'{v}_pools.npz') as f:pool=dict(f)
        with np.load(Path(cfg['parent_root'])/f'{v}_pools.npz') as f:parent=dict(f)
        for pe in range(3):
            for i in range(24):
                for j in range(24):
                    n=parent['counts'][pe,i,j];assert pool['counts'][pe,i,j]>=n and np.array_equal(pool['transforms'][pe,i,j,:n],parent['transforms'][pe,i,j,:n]);checks+=1
        with np.load(root/f'{v}_fits.npz') as f:fits=dict(f)
        with np.load(root/f'{v}_cache.npz') as f:cache=dict(f)
        with np.load(root/f'{v}_search_graph.npz') as f:graph=dict(f)
        lookup={tuple(t):h for h,t in enumerate(cache['triples'])};fitlookup={tuple(t):h for h,t in enumerate(fits['triples'])}
        for fi in range(len(fits['triples'])):
            for n in range(int(fits[f'{fi}_count'])):
                path=fits[f'{fi}_{n}_path'];assert np.max(np.diff(path),initial=0)<=1e-10;fitcount+=1;improvedfits+=int(path[-1]<path[0]-1e-10);checks+=1
        for s,val in zip(graph['states'],graph['values']):
            assert all(np.array_equal(np.sort(s[:,e]),np.arange(24)) for e in range(2));rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(4));assert np.max(abs(val-rebuilt))<1e-9;checks+=1
        oracle=[]
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);A=pool['transforms'][0,i,j,:pool['counts'][0,i,j]];B=pool['transforms'][1,i,k,:pool['counts'][1,i,k]];z1=np.einsum('bij,nj->bni',A[:,:3,:3],x)+A[:,None,:3,3];z2=np.einsum('bij,nj->bni',B[:,:3,:3],x)+B[:,None,:3,3];a=np.quantile(np.linalg.norm(z1-y1,axis=2),.95,axis=1);b=np.quantile(np.linalg.norm(z2-y2,axis=2),.95,axis=1);c=np.quantile(np.linalg.norm((z2[None]-z1[:,None])-(y2-y1),axis=3),.95,axis=2);base=float(np.maximum(np.maximum(a[:,None],b[None]),c).min());best=base;fi=fitlookup.get((i,j,k))
            if fi is not None:
                for n in range(int(fits[f'{fi}_count'])):best=min(best,float(physical_error(*fits[f'{fi}_{n}_T'],x,y1,y2)))
            oracle.append((base,best))
        for col,m in enumerate(METHODS):
            with np.load(root/f'{v}_{m}.npz') as f:pred=dict(f)
            assert int(pred['search_state'])==graph['values'][:,col].argmin();checks+=1
            rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc)
            for i,(j,k) in enumerate(pred['target_slots']):
                j,k=int(j),int(k);h=lookup[i,j,k];T=pred['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);keys=[(0,i,j),(1,i,k),(2,j,k)];Ts=[T[0],T[1],T[1]@inverse(T[0])];tr=[PairTransport(banks[e][a],banks[f][b],edges[pe][a*24+b],views[e][a],views[f][b],a,b,cfg) for (pe,a,b),(e,f) in zip(keys,PAIRS)];ev=[p.evidence(U) for p,U in zip(tr,Ts)];vv,state=joint_values(ev,cfg,True)
                score=sum(w*make_scorer(pe,a,b,ep,groups,views,edges,bc).score(U[None])[0] for w,(pe,a,b),U in zip([1.,1.,.5],keys,Ts)) if col==0 else vv[col-1];max_error=max(max_error,abs(float(score)-float(pred['objectives'][i])));assert abs(score-pred['objectives'][i])<1e-8;checks+=1
                origin=int(cache['origins'][h,col]);producer=-1
                if origin>=0:fi,n=divmod(origin,100);producer=int(fits[f'{fi}_{n}_producer']);assert np.max(abs(T-fits[f'{fi}_{n}_T']))<1e-12;checks+=1
                rr[i].update(rear=int(perm[2,i]%3==0),oracle_discrete_error_m=oracle[i][0],oracle_pool_error_m=oracle[i][1],oracle_pool_qualified=int(oracle[i][1]<=.05),selected_new_endpoint=int(origin>=0),endpoint_producer=producer,correct_identity_tuple_searched=int((i,int(perm[1,i]),int(perm[2,i])) in lookup))
                assert rr[i]['qualified']<=rr[i]['oracle_pool_qualified'];checks+=1
                if col:
                    chosen=selected_edges(ev,state,m,[banks[0][i],banks[1][j],banks[2][k]]);match_count=0;correct=0
                    for r,(e,f),(pe,a,b) in zip(chosen,PAIRS,keys):
                        ai=int(np.flatnonzero(perm[e]==a)[0]);bi=int(np.flatnonzero(perm[f]==b)[0]);xa=apply(inverse(poses[e,ai]),r['x']);yb=apply(inverse(poses[f,bi]),r['y']);match_count+=len(xa);correct+=int((np.linalg.norm(xa-yb,axis=1)<=.015).sum()) if ai==bi else 0
                    clusters=state['memory' if m=='history_union' else 'shared'];diagnostics.append(dict(variant=v,method=m,asset=i,both_ids=rr[i]['both_ids'],rear=rr[i]['rear'],matches=match_count,correct_matches_15mm=correct,shared_partition_gain=float(state['shared_path'][-1]-state['memory_path'][-1]),two_epoch_tracks=int(((clusters>=0).sum(1)==2).sum()),three_epoch_tracks=int(((clusters>=0).sum(1)==3).sum()),energy=score))
            rows+=rr
            # Separate discrete-only stage; same finite identity graph.
            q=int(graph['discrete_values'][:,col].argmin());s=graph['states'][q];p=dict(target_slots=s,transforms=np.array([cache['discrete_transforms'][lookup[i,int(j),int(k)],col] for i,(j,k) in enumerate(s)]),triplet_counts=np.zeros(24,int));rr=task_rows(ic['scene'],'train',v,m+'_discrete',p,truth,anchors,bc)
            for i,r in enumerate(rr):r.update(rear=int(perm[2,i]%3==0),oracle_discrete_error_m=oracle[i][0],oracle_pool_error_m=oracle[i][0],oracle_pool_qualified=int(oracle[i][0]<=.05),selected_new_endpoint=0,endpoint_producer=-1,correct_identity_tuple_searched=int((i,int(perm[1,i]),int(perm[2,i])) in lookup))
            rows+=rr
        with np.load(Path(cfg['parent_root'])/f'{v}_native_ransac.npz') as f:p=dict(f)
        rr=task_rows(ic['scene'],'train',v,'previous_native_ransac',p,truth,anchors,bc)
        for i,r in enumerate(rr):r.update(rear=int(perm[2,i]%3==0),oracle_discrete_error_m=None,oracle_pool_error_m=None,oracle_pool_qualified=None,selected_new_endpoint=0,endpoint_producer=-1,correct_identity_tuple_searched=None)
        rows+=rr
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'material_diagnostics.csv',diagnostics);summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS+[m+'_discrete' for m in METHODS]+['previous_native_ransac']:
            rr=[r for r in rows if r['variant']==v and r['method']==m];good=[r for r in rr if r['both_ids']];summary.append(dict(variant=v,method=m,tasks=len(rr),qualified=sum(r['qualified'] for r in rr),both_ids=len(good),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),front_qualified=sum(r['qualified'] for r in rr if not r['rear']),oracle_pool_qualified=sum(r['oracle_pool_qualified'] or 0 for r in rr) if m!='previous_native_ransac' else None,selected_new_endpoints=sum(r['selected_new_endpoint'] for r in rr),correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in good])) if good else None))
        for base,cand in [('previous_native_ransac','ordinary_union'),('ordinary_union','shared_transport'),('visibility_pair','shared_transport'),('history_union','shared_transport')]+[(m+'_discrete',m) for m in METHODS]:
            a={r['asset']:r for r in rows if r['variant']==v and r['method']==base};b={r['asset']:r for r in rows if r['variant']==v and r['method']==cand};common=[i for i in a if a[i]['both_ids'] and b[i]['both_ids']];comparisons.append(dict(variant=v,baseline=base,candidate=cand,gained=[i for i in a if b[i]['qualified'] and not a[i]['qualified']],lost=[i for i in a if a[i]['qualified'] and not b[i]['qualified']],common_correct_ids=len(common),common_correct_id_worst_error_change_median_m=float(np.median([b[i]['worst_error_m']-a[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(summary=summary,comparisons=comparisons,checks=checks,primary_rows=384,total_rows=len(rows),fit_count=fitcount,fits_objective_improved=improvedfits,max_reconstruction_error=max_error,scope='Single TRAIN scene, 24 correlated objects x 4 correlated appearances; no inferential significance or generalization claim. Matching diagnostic uses private poses to map measured XYZ to approximate material coordinates; not supervision in inference.'))
    print(json.dumps(dict(checks=checks,primary_rows=384,fit_count=fitcount,max_reconstruction_error=max_error,summary=[r for r in summary if r['method'] in METHODS]),indent=2))

if __name__=='__main__':main()
