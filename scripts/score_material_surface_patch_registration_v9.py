"""Post-seal full task scores, generator-specific oracle, raw pixel provenance."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv,argparse
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_matching_v1 import apply,world_points,rigid
from scipy.spatial import cKDTree
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from scripts.run_material_surface_patch_registration_v9 import METHODS,setup,make_all,PAIRS
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save,sha

def pool_error(A,B,x,y1,y2):
    a=np.einsum('bij,nj->bni',A[:,:3,:3],x)+A[:,None,:3,3];b=np.einsum('bij,nj->bni',B[:,:3,:3],x)+B[:,None,:3,3];e1=np.quantile(np.linalg.norm(a-y1,axis=2),.95,axis=1);e2=np.quantile(np.linalg.norm(b-y2,axis=2),.95,axis=1);e3=np.quantile(np.linalg.norm((b[None]-a[:,None])-(y2-y1),axis=3),.95,axis=2);return float(np.maximum(np.maximum(e1[:,None],e2[None]),e3).min())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/material_surface_patch_registration_v9_run2');args=ap.parse_args();cfg=json.loads(Path('configs/material_surface_patch_registration_v9.json').read_text(encoding='utf-8'));root=Path(args.root);parent=Path(cfg['parent_root']);geom=Path(cfg['geometry_cache']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    ss=json.loads((geom/'surface_seal.json').read_text(encoding='utf-8'))
    for p,h in ss['sources'].items():assert sha(p)==h;checks+=1
    for p,h in ss['files'].items():assert sha(geom/p)==h;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(scene);poses=np.array(truth['poses']);perm=np.array(truth['permutations']);rows=[];oracle_rows=[];pruning=[];true_pose_costs=[];max_error=0.;raw_points=0;raw_patch_samples=0;fit_counts=[0,0,0]
    for v in cfg['variants']:
        _,frames,ids,bc,ep,groups,views,banks,edges=setup(cfg,v)
        # Raw provenance uses public RGBD only, not the scoring labels above.
        for e in range(3):
            for slot,a in enumerate(banks[e]):
                for fi in np.unique(a['frame']):
                    ix=a['frame']==fi;p=a['uv'][ix];f=frames[e][int(fi)];assert np.max(abs(world_points(f,p)-a['xyz'][ix]))<1e-12 and np.array_equal(f['rgb'][p[:,1],p[:,0]]/255.,a['rgb'][ix]);assert (f['anonymous_slot'][p[:,1],p[:,0]]==slot).all();raw_points+=len(p);checks+=3
                for h,fi in enumerate(a['patch_frame']):
                    f=frames[e][int(fi)];p=a['patch_uv'][h];assert np.max(abs(world_points(f,p)-a['patch_xyz'][h]))<1e-12 and np.array_equal(f['rgb'][p[:,1],p[:,0]]/255.,a['patch_rgb'][h]);assert (f['anonymous_slot'][p[:,1],p[:,0]]==slot).all();raw_patch_samples+=len(p);checks+=3
        with np.load(root/f'{v}_pools.npz') as f:pool=dict(f)
        with np.load(parent/f'{v}_pools.npz') as f:oldpool=dict(f)
        with np.load(root/f'{v}_fits.npz') as f:fits=dict(f)
        with np.load(root/f'{v}_cache.npz') as f:cache=dict(f)
        with np.load(parent/f'{v}_cache.npz') as f:pc=dict(f)
        with np.load(root/f'{v}_search_graph.npz') as f:graph=dict(f)
        with np.load(parent/f'{v}_search_graph.npz') as f:pg=dict(f)
        for key in np.ndindex(3,24,24):
            n=int(oldpool['counts'][key]);assert pool['counts'][key]>=n and np.array_equal(pool['transforms'][key][:n],oldpool['transforms'][key][:n]);checks+=1
        gstate={s.tobytes() for s in graph['states']};assert all(s.tobytes() in gstate for s in pg['states']);checks+=1
        assert graph['values'][:,0].min()<=pg['values'][:,0].min()+1e-8;checks+=1
        for key,path in fits.items():
            if not key.endswith('_path'):continue
            kind=int(key.split('_')[-2]);fit_counts[kind]+=1
            if kind==2:assert np.max(np.diff(path),initial=0)<=1e-12;checks+=1
        lookup={tuple(t):h for h,t in enumerate(cache['triples'])};plook={tuple(t):h for h,t in enumerate(pc['triples'])}
        for s,val in zip(graph['states'],graph['values']):
            assert all(np.array_equal(np.sort(s[:,e]),np.arange(24)) for e in range(2));rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(5));assert np.max(abs(rebuilt-val))<1e-8;checks+=1
        globals=[]
        for pe in range(3):
            with np.load(geom/f'pair{pe}_geometry.npz') as f:globals.append(f['transforms'])
        oracle=[];gt_scores=[]
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);paired=pc['transforms'][plook[i,j,k]] if (i,j,k) in plook else []
            families={}
            for family in ['old','fgr','icp','colored','projective','union']:
                AB=[]
                for pe,target in [(0,j),(1,k)]:
                    n=int(oldpool['counts'][pe,i,target]);A=list(oldpool['transforms'][pe,i,target,:n])
                    if family!='old':A+=list(globals[pe][i,target])
                    if family in ['icp','colored','projective']:
                        kind={'icp':0,'colored':1,'projective':2}[family]
                        A += [fits[key] for key in fits if key.startswith(f'{pe}_{i}_{target}_') and key.endswith(f'_{kind}_T')]
                    if family=='union':A=list(pool['transforms'][pe,i,target,:pool['counts'][pe,i,target]])
                    AB.append(np.array(A))
                value=pool_error(*AB,x,y1,y2)
                for T in paired:value=min(value,float(physical_error(*T,x,y1,y2)))
                families[family]=value
            gtT=[poses[1,i]@inverse(poses[0,i]),poses[2,i]@inverse(poses[0,i])];gtU=[*gtT,gtT[1]@inverse(gtT[0])];score=sum(w*make_all(pe,a,b,bc,ep,groups,views,banks,edges,cfg).score(U[None])[0] for w,(pe,a,b),U in zip([1,1,.5],[(0,i,j),(1,i,k),(2,j,k)],gtU));gt_scores.append(score)
            # Label-assisted association oracle is a diagnostic, not observability proof.
            local=[apply(inverse(poses[e,i]),banks[e][slot]['xyz']) for e,slot in enumerate([i,j,k])];association={}
            for radius in [.006,.015]:
                AB=[];counts=[]
                for e,target in [(1,j),(2,k)]:
                    dist,idx=cKDTree(local[e]).query(local[0]);rev=cKDTree(local[0]).query(local[e])[1];ok=(dist<=radius)&(rev[idx]==np.arange(len(local[0])));source=np.flatnonzero(ok);counts.append(len(source));T=rigid(banks[0][i]['xyz'][source],banks[e][target]['xyz'][idx[source]]);AB.append(T)
                value=float(physical_error(*AB,x,y1,y2)) if all(T is not None for T in AB) else None;tag='6mm' if radius==.006 else '15mm';association[f'association_{tag}_error_m']=value;association[f'association_{tag}_h1_points']=counts[0];association[f'association_{tag}_h2_points']=counts[1]
            oracle.append(families);oracle_rows.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),correct_identity_searched=int((i,j,k) in lookup),**{f'{a}_error_m':b for a,b in families.items()},**association,**{f'true_pose_energy_{m}':float(s) for m,s in zip(METHODS,score)}))
        true_pose_costs.append(dict(variant=v,true_global_assignment_energy=np.array(gt_scores).sum(0).tolist(),selected_global_assignment_energy=graph['values'].min(0).tolist(),meaning='Private poses/IDs used only after prediction seal; energy ceiling diagnostic, not new predictions.'))
        for col,m in enumerate(METHODS):
            with np.load(root/f'{v}_{m}.npz') as f:pred=dict(f)
            state=int(pred['search_state']);assert state==graph['values'][:,col].argmin() and np.array_equal(pred['target_slots'],graph['states'][state]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc)
            for i,(j,k) in enumerate(pred['target_slots']):
                j,k=int(j),int(k);h=lookup[i,j,k];T=pred['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);sc=[make_all(pe,a,b,bc,ep,groups,views,banks,edges,cfg) for pe,a,b in [(0,i,j),(1,i,k),(2,j,k)]];U=[T[0],T[1],T[1]@inverse(T[0])];value=sum(w*s.score(t[None])[0,col] for w,s,t in zip([1,1,.5],sc,U));err=abs(value-pred['objectives'][i]);max_error=max(max_error,float(err));assert err<1e-8;checks+=1
                rr[i].update(rear=int(perm[2,i]%3==0),oracle_pool_qualified=int(oracle[i]['union']<=.05),oracle_error_m=oracle[i]['union'],old_oracle_error_m=oracle[i]['old']);assert rr[i]['qualified']<=rr[i]['oracle_pool_qualified'];checks+=1
            rows+=rr
        # Independent brute-force check of the nonnegative cost pruning.
        with np.load(parent/f'{v}_shared_transport.npz') as f:previous=dict(f)
        i=0;j,k=map(int,previous['target_slots'][0]);A=pool['transforms'][0,i,j,:pool['counts'][0,i,j]];B=pool['transforms'][1,i,k,:pool['counts'][1,i,k]];T1=np.repeat(A,len(B),0);T2=np.tile(B,(len(A),1,1));s0=make_all(0,i,j,bc,ep,groups,views,banks,edges,cfg);s1=make_all(1,i,k,bc,ep,groups,views,banks,edges,cfg);s2=make_all(2,j,k,bc,ep,groups,views,banks,edges,cfg);full=s0.score(A)[:,None]+s1.score(B)[None]+.5*s2.score(T2@inverse(T1)).reshape(len(A),len(B),5);minimum=full.reshape(-1,5).min(0)
        if (i,j,k) in plook:
            T=pc['transforms'][plook[i,j,k]];minimum=np.minimum(minimum,(s0.score(T[:,0])+s1.score(T[:,1])+.5*s2.score(T[:,1]@inverse(T[:,0]))).min(0))
        err=float(np.max(abs(minimum-cache['costs'][lookup[i,j,k]])));assert err<1e-8;checks+=1;pruning.append(dict(variant=v,pose_pairs=len(T1),max_error=err))
        for previous_method in ['ordinary_union','shared_transport']:
            with np.load(parent/f'{v}_{previous_method}.npz') as f:p=dict(f)
            rr=task_rows(ic['scene'],'train',v,'previous_'+previous_method,p,truth,anchors,bc)
            for i,r in enumerate(rr):r.update(rear=int(perm[2,i]%3==0),oracle_pool_qualified=int(oracle[i]['old']<=.05),oracle_error_m=oracle[i]['old'],old_oracle_error_m=oracle[i]['old'])
            rows+=rr
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_by_producer.csv',oracle_rows);summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS+['previous_ordinary_union','previous_shared_transport']:
            rr=[r for r in rows if r['variant']==v and r['method']==m];good=[r for r in rr if r['both_ids']];summary.append(dict(variant=v,method=m,tasks=len(rr),qualified=sum(r['qualified'] for r in rr),both_ids=len(good),front_qualified=sum(r['qualified'] for r in rr if not r['rear']),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_pool_qualified=sum(r['oracle_pool_qualified'] for r in rr),correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in good])) if good else None))
        for base,cand in [('previous_shared_transport','ordinary_native'),('ordinary_native','material_patch'),('surface_geometry','material_patch'),('surface_rgb_fusion','material_patch'),('projective_points','material_patch')]:
            a={r['asset']:r for r in rows if r['variant']==v and r['method']==base};b={r['asset']:r for r in rows if r['variant']==v and r['method']==cand};common=[i for i in a if a[i]['both_ids'] and b[i]['both_ids']];comparisons.append(dict(variant=v,baseline=base,candidate=cand,gained=[i for i in a if b[i]['qualified'] and not a[i]['qualified']],lost=[i for i in a if a[i]['qualified'] and not b[i]['qualified']],common_correct_ids=len(common),common_correct_id_worst_error_change_median_m=float(np.median([b[i]['worst_error_m']-a[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(summary=summary,comparisons=comparisons,checks=checks,primary_rows=480,total_rows=len(rows),fit_counts=fit_counts,raw_points_verified=raw_points,raw_patch_samples_verified=raw_patch_samples,max_selected_objective_error=max_error,pruning_checks=pruning,true_pose_costs=true_pose_costs,scope='One outcome-informed TRAIN scene; 24 objects/4 appearances correlated. Official Open3D ColoredICP is a candidate generator, not a standalone full published system comparison. No new training or independent confirmation.'))
    print(json.dumps(dict(primary_rows=480,checks=checks,fit_counts=fit_counts,summary=summary,comparisons=comparisons),indent=2))

if __name__=='__main__':main()
