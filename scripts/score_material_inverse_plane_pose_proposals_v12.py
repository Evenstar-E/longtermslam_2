"""Only after full prediction seal: full tasks, stage oracles, exact replay."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration,METHODS,PAIRS,setup
from inspection_decisions.material_oriented_surface_control_v10 import OrientedScore,orient_banks,inverse
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import TripleScore
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_material_surface_patch_registration_v9 import pool_error
from scripts.score_material_native_modes_v6 import physical_error
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def load(path):
    with np.load(path) as f:return dict(f)


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene)
    rows=[];oracles=[];brute=[];mechanism=[];max_error=0.;fit_count=0
    for v in cfg['variants']:
        _,frames,_,bc,_,_,views,banks,_=setup(cfg,v);orient_banks(banks,frames,cfg)
        cache=load(root/f'{v}_cache.npz');graph=load(root/f'{v}_search_graph.npz');proposals=load(root/f'{v}_proposals.npz');fits=load(root/f'{v}_fits.npz');old=load(Path(cfg['retained_root'])/f'{v}_cache.npz');oldgraph=load(Path(cfg['retained_root'])/f'{v}_search_graph.npz');pools=load(Path(cfg['parent_root'])/f'{v}_pools.npz')
        assert np.array_equal(cache['triples'],old['triples']) and np.array_equal(graph['states'],oldgraph['states']);assert np.max(abs(cache['costs'][:,:2]-old['costs'][:,2:4]))<1e-12;checks+=2
        assert (cache['costs'][:,2:4]<=cache['costs'][:,:2]+1e-10).all() and (cache['costs'][:,4:6]<=cache['costs'][:,2:4]+1e-10).all();checks+=1
        lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
        def scorer(i,j,k):return TripleScore([OrientedScore(banks[e][a],banks[f][b],views[e][a],views[f][b],a,b,cfg) for (e,f),a,b in zip(PAIRS,[i,i,j],[j,k,k])],cfg)
        for key,path in fits.items():
            if key.endswith('_path'):assert np.max(np.diff(path),initial=0)<=1e-12;fit_count+=1;checks+=1
        for s,val in zip(graph['states'],graph['values']):
            assert all(np.array_equal(np.sort(s[:,e]),np.arange(24)) for e in range(2));rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8));assert np.max(abs(rebuilt-val))<1e-8;checks+=2
        # Every saved new endpoint has a rigid transform and recomputable score.
        for h,triple in enumerate(cache['triples']):
            U=proposals[f'{h}_transforms'];R=U[:,:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8 and np.isfinite(U).all();checks+=1
            assert all(any(np.max(abs(T-Q))<1e-9 for Q in U) for T in old['transforms'][h]);checks+=1
        stage_oracle={}
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
            A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]];err=pool_error(A,B,x,y1,y2)
            stages=dict(retained=err,propagated=err,continuous=err)
            if (i,j,k) in lookup:
                h=lookup[i,j,k];U=proposals[f'{h}_transforms'];stage=proposals[f'{h}_stage'];errors=np.array([float(physical_error(*T,x,y1,y2)) for T in U])
                for name,limit in [('retained',0),('propagated',1),('continuous',4)]:
                    if (stage<=limit).any():stages[name]=min(err,float(errors[stage<=limit].min()))
                best=int(errors.argmin());GT=np.array([poses[e,i]@inverse(poses[0,i]) for e in [1,2]])
                gtvalue=scorer(i,j,k).score(GT[None])[0];bv=proposals[f'{h}_values'][best]
                mechanism.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),best_new_error_m=float(errors[best]),best_new_stage=int(stage[best]),new_candidates=len(U),true_point=float(gtvalue[0]),true_patch=float(gtvalue[1]),true_independent=float(gtvalue[2]),true_shared=float(gtvalue[3]),best_pose_point=float(bv[0]),best_pose_patch=float(bv[1]),best_pose_independent=float(bv[2]),best_pose_shared=float(bv[3])))
            stage_oracle[i]=stages;oracles.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),correct_identity_in_graph=int((i,j,k) in lookup),**{f'{name}_error_m':e for name,e in stages.items()}))
        for col,m in enumerate(METHODS):
            pred=load(root/f'{v}_{m}.npz');s=int(pred['search_state']);assert s==int(graph['best'][col]) and np.array_equal(pred['target_slots'],graph['states'][s]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc)
            for i,(j,k) in enumerate(pred['target_slots']):
                j,k=int(j),int(k);h=lookup[i,j,k];T=pred['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);value=scorer(i,j,k).score(T[None])[0,[0,1,0,1,0,1,2,3][col]];error=abs(float(value)-float(pred['objectives'][i]));max_error=max(max_error,error);assert error<1e-8;checks+=2
                stage='retained' if col<2 else 'propagated' if col<4 else 'continuous';o=stage_oracle[i][stage];rr[i].update(rear=int(perm[2,i]%3==0),oracle_error_m=o,oracle_qualified=int(o<=.05),selected_origin=int(cache['origins'][h,col]));assert rr[i]['qualified']<=rr[i]['oracle_qualified'];checks+=1
            rows+=rr
        # One complete Cartesian brute force per appearance, independent of pruning.
        i,j,k=map(int,cache['triples'][0]);A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]]
        T=np.stack([np.repeat(A,len(B),0),np.tile(B,(len(A),1,1))],1);sc=scorer(i,j,k);full=sc.score(T);saved=proposals['0_values'];minimum=np.minimum(full[:,2:].min(0),saved[:,2:].min(0));error=float(np.max(abs(minimum-cache['costs'][0,6:])));assert error<1e-8;checks+=1;brute.append(dict(variant=v,pose_pairs=len(T),max_error=error))
        for col,m in enumerate(['normal_points','normal_patch']):
            p=load(Path(cfg['retained_root'])/f'{v}_{m}.npz');new=load(root/f'{v}_{METHODS[col]}.npz');assert np.array_equal(p['target_slots'],new['target_slots']) and np.array_equal(p['transforms'],new['transforms']);checks+=1
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_by_stage.csv',oracles);write_csv(root/'pose_mechanism.csv',mechanism)
    summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS:
            rr=[r for r in rows if r['variant']==v and r['method']==m];good=[r for r in rr if r['both_ids']]
            summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=len(good),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_qualified=sum(r['oracle_qualified'] for r in rr),correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in good])) if good else None))
        for a,b in [('retained_points','propagated_points'),('retained_patch','propagated_patch'),('propagated_points','continuous_points'),('propagated_patch','continuous_patch'),('continuous_patch','independent_material'),('independent_material','shared_material')]:
            aa={r['asset']:r for r in rows if r['variant']==v and r['method']==a};bb={r['asset']:r for r in rows if r['variant']==v and r['method']==b};common=[i for i in aa if aa[i]['both_ids'] and bb[i]['both_ids']]
            comparisons.append(dict(variant=v,baseline=a,candidate=b,gained=[i for i in aa if bb[i]['qualified'] and not aa[i]['qualified']],lost=[i for i in aa if aa[i]['qualified'] and not bb[i]['qualified']],common_correct_ids=len(common),paired_error_change_median_m=float(np.median([bb[i]['worst_error_m']-aa[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(primary_rows=len(rows),summary=summary,comparisons=comparisons,checks=checks,fit_paths_verified=fit_count,max_selected_objective_error=max_error,brute_force=brute,scope=cfg['contract']));print(json.dumps(dict(checks=checks,summary=summary,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
