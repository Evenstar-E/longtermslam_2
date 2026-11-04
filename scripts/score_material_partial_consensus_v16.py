"""Private anchor evaluation only after ALL four appearances are sealed."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scripts.run_material_partial_consensus_v16 import configuration,METHODS,SCORE_COLUMNS,PAIRS,load,CachedPair,Triple,FAMILIES
from scripts.prepare_material_surface_fields_v15 import load_fields
from inspection_decisions.material_continuous_surface_cache_v14b import DensePair
from inspection_decisions.material_continuous_exposure_v14c import directed_match
from inspection_decisions.material_surface_distribution_v15b import dense_modes,RegionPair
from scripts.run_material_continuous_surface_correspondence_v14c import setup,dense_records
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from inspection_decisions.textured_material_matching_v1 import apply,world_points
from scripts.run_textured_material_comparison_v1 import save,sha


def job(c,v):
    import torch
    torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path(c['output']);ic,frames,ids,bc,_,_,views,banks,_=setup(c,v);dense=dense_records(views);load_fields(c,v,banks,dense);scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);cache=load(root/f'{v}_cache.npz');pool=load(root/f'{v}_proposals.npz');graph=load(root/f'{v}_search_graph.npz');parent=Path(c['parent_root']);old=load(parent/f'{v}_cache.npz');oldpool=load(parent/f'{v}_proposals.npz');oldgraph=load(parent/f'{v}_search_graph.npz');checks=0;centers=0;maximum=0.;rows=[];oracle=[];failure=[];lookup={tuple(t):h for h,t in enumerate(cache['triples'])};names=['retained']+FAMILIES+['union']
    assert np.array_equal(cache['triples'],old['triples']) and np.array_equal(graph['states'],oldgraph['states']);checks+=1
    for e in range(3):
        for a in banks[e]:
            assert len(a['xyz'])==512;centers+=len(a['xyz']);checks+=1
            for fi in np.unique(a['frame']):
                h=a['frame']==fi;assert np.max(abs(world_points(frames[e][int(fi)],a['uv'][h])-a['xyz'][h]))<1e-10;checks+=1
    for h,triple in enumerate(cache['triples']):
        U=pool[f'{h}_transforms'];retained=oldpool[f'{h}_transforms'];assert len(U)>=len(retained) and np.array_equal(U[:len(retained)],retained);checks+=1
        assert np.array_equal(pool[f'{h}_values'][:len(retained)],oldpool[f'{h}_values'][:,SCORE_COLUMNS]);checks+=1
        R=U[:,:,:3,:3];assert np.isfinite(U).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
        vv=pool[f'{h}_values'];eligible=pool[f'{h}_family_eligible'];cost=np.r_[old['costs'][h,5],vv.min(0),[vv[eligible[:,f],2].min() for f in range(4)]];assert np.max(abs(cost-cache['costs'][h]))<1e-8 and eligible[:len(retained)].all();checks+=1
    for state,value in zip(graph['states'],graph['values']):
        assert all(np.array_equal(np.sort(state[:,e]),np.arange(24)) for e in range(2));vv=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(8));assert np.max(abs(vv-value))<1e-8;checks+=2
    @lru_cache(maxsize=128)
    def pair(pe,i,j):
        e,f=PAIRS[pe];a={**banks[e][i],'normalizer':banks[e][i][f'normalizer_{f}']};b={**banks[f][j],'normalizer':banks[f][j][f'normalizer_{e}']};base=DensePair(a,b,views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],c);return CachedPair(RegionPair(base,dense_modes(a,dense[f][j],c),dense_modes(b,dense[e][i],c),c))
    oracles={};errors={}
    for i,a in enumerate(anchors):
        j,k=map(int,perm[1:,i]);h=lookup.get((i,j,k));data={name:None for name in names}
        if h is not None:
            x=apply(poses[0,i],a);y1=apply(poses[1,i],a);y2=apply(poses[2,i],a);U=pool[f'{h}_transforms'];E=np.array([physical_error(*T,x,y1,y2) for T in U]);n=int(pool[f'{h}_retained_count']);eligible=pool[f'{h}_family_eligible'];masks=[np.arange(len(U))<n]+[eligible[:,f] for f in range(4)]+[np.ones(len(U),bool)]
            for name,mask in zip(names,masks):data[name]=float(E[mask].min())
            errors[i]=E
        oracles[i]=data;oracle.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),true_identity_visited=int(h is not None),**{name+'_error_m':e for name,e in data.items()}))
    for col,m in enumerate(METHODS):
        pred=load(root/f'{v}_{m}.npz');state=int(pred['search_state']);assert state==int(graph['best'][col]) and np.array_equal(pred['target_slots'],graph['states'][state]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc);oname='retained' if col==0 else 'union' if col<4 else FAMILIES[col-4]
        for i,(j,k) in enumerate(pred['target_slots']):
            j,k=int(j),int(k);h=lookup[i,j,k];T=pred['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);checks+=1
            if col>0:
                sc=Triple([pair(0,i,j),pair(1,i,k),pair(2,j,k)]);vv=sc.score(T[None])[0];scorecol=SCORE_COLUMNS[col-1] if col<4 else 3;error=abs(float(vv[scorecol])-float(pred['objectives'][i]));assert error<1e-8,(v,m,i,error);maximum=max(maximum,error);checks+=1
            oo=oracles[i][oname];rr[i].update(rear=int(perm[2,i]%3==0),oracle_error_m=oo,oracle_qualified=int(oo is not None and oo<=.05));assert rr[i]['qualified']<=rr[i]['oracle_qualified'];checks+=1
            if col>0:
                tj,tk=map(int,perm[1:,i]);qh=lookup.get((i,tj,tk));best=oracles[i][oname]
                if rr[i]['qualified']:category='success'
                elif qh is None:category='true_identity_not_visited'
                elif best>.05:category='no_qualified_pose_in_true_identity_pool'
                elif rr[i]['both_ids']:category='objective_prefers_bad_pose_at_correct_identity'
                else:category='identity_assignment_failed_despite_qualified_pose'
                failure.append(dict(variant=v,method=m,asset=i,rear=rr[i]['rear'],category=category,selected_error_m=rr[i]['worst_error_m'],pool_best_error_m=best))
        rows+=rr
        if col==0:
            p=load(parent/f'{v}_common_point_density.npz');assert np.array_equal(p['target_slots'],pred['target_slots']) and np.array_equal(p['transforms'],pred['transforms']);checks+=1
    i,j,k=map(int,cache['triples'][0]);values=Triple([pair(0,i,j),pair(1,i,k),pair(2,j,k)]).score(pool['0_transforms'])[:,SCORE_COLUMNS];assert np.max(abs(values-pool['0_values']))<1e-8;checks+=1
    trace=load(root/f'{v}_pair_proposals.npz');paths=0;starts=0
    for key,value in trace.items():
        if key.endswith('_path'):assert np.isfinite(value).all() and (np.diff(value)<=1e-10).all() and len(value)<=c['fit_steps']+1;paths+=1
        if 'local3_tls_' in key and key.endswith('_initial'):assert np.array_equal(value,trace[key.replace('local3_tls_','local3_partial_')]);starts+=1
    checks+=paths+starts;print('V16 SCORE',v,checks,flush=True);return rows,oracle,failure,checks,centers,maximum,paths,starts


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    rows=[];oracle=[];failure=[];centers=0;maximum=0.;paths=0;starts=0
    with ProcessPoolExecutor(max_workers=c['workers']) as executor:
        for f in as_completed([executor.submit(job,c,v) for v in c['variants']]):
            rr,oo,ff,n,nc,err,pa,st=f.result();rows+=rr;oracle+=oo;failure+=ff;checks+=n;centers+=nc;maximum=max(maximum,err);paths+=pa;starts+=st
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_by_family.csv',oracle);write_csv(root/'failure_attribution.csv',failure);summary=[];comparisons=[]
    for v in c['variants']:
        for m in METHODS:
            rr=[r for r in rows if r['variant']==v and r['method']==m];good=[r for r in rr if r['both_ids']];summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=len(good),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_qualified=sum(r['oracle_qualified'] for r in rr),correct_id_median_error_m=float(np.median([r['worst_error_m'] for r in good])) if good else None))
        for a,b in [('retained_v15_point','common_point_density'),('common_geometry','common_point_density'),('common_descriptor_fusion','common_point_density'),('retained_v15_point','family_global'),('family_global','family_local'),('family_local','family_triplet'),('family_triplet','family_partial')]:
            aa={r['asset']:r for r in rows if r['variant']==v and r['method']==a};bb={r['asset']:r for r in rows if r['variant']==v and r['method']==b};common=[i for i in aa if aa[i]['both_ids'] and bb[i]['both_ids']];comparisons.append(dict(variant=v,baseline=a,candidate=b,gained=[i for i in aa if bb[i]['qualified'] and not aa[i]['qualified']],lost=[i for i in aa if aa[i]['qualified'] and not bb[i]['qualified']],common_correct_ids=len(common),paired_worst_error_change_median_m=float(np.median([bb[i]['worst_error_m']-aa[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(primary_rows=len(rows),checks=checks,source_centers_verified=centers,max_selected_objective_error=maximum,monotone_paths_verified=paths,identical_triplet_starts_verified=starts,summary=summary,comparisons=comparisons,scope=c['contract_v16']));print(json.dumps(dict(checks=checks,summary=summary,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
