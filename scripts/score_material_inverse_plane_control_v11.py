"""Post-seal full tasks and pose-matched normal-filter counterfactuals."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_material_inverse_plane_control_v11 import configuration,METHODS,PAIRS,setup
from inspection_decisions.material_oriented_surface_control_v10 import OrientedScore,orient_banks,inverse
from inspection_decisions.material_inverse_plane_control_v11 import project
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    cfg=configuration();root=Path(cfg['output']);parent=Path(cfg['parent_root']);s=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert s['complete'] and not s['private_labels_read'];checks=1
    for p,h in s['sources'].items():assert sha(p)==h;checks+=1
    for p,h in s['files'].items():assert sha(root/p)==h;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);oldrows=list(csv.DictReader((parent/'task_metrics.csv').open(encoding='utf-8-sig')));rows=[];counter=[];normals=[];brute=[];energy=[];max_error=0.
    for v in cfg['variants']:
        _,frames,_,bc,ep,groups,views,banks,edges=setup(cfg,v);orient_banks(banks,frames,cfg)
        with np.load(root/f'{v}_cache.npz') as f:cache=dict(f)
        with np.load(root/f'{v}_search_graph.npz') as f:graph=dict(f)
        with np.load(parent/f'{v}_cache.npz') as f:oldcache=dict(f)
        with np.load(parent/f'{v}_search_graph.npz') as f:oldgraph=dict(f)
        assert np.array_equal(cache['triples'],oldcache['triples']) and np.array_equal(graph['states'],oldgraph['states']);checks+=3;lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
        def score(key,T):
            i,j,k=map(int,key);value=np.zeros(4)
            for w,pe,a,b,U in [(1,0,i,j,T[0]),(1,1,i,k,T[1]),(.5,2,j,k,T[1]@inverse(T[0]))]:
                e,f=PAIRS[pe];value+=w*OrientedScore(banks[e][a],banks[f][b],views[e][a],views[f][b],a,b,cfg).score(U[None])[0]
            return value
        for state,val in zip(graph['states'],graph['values']):
            rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(4));assert np.max(abs(rebuilt-val))<1e-8;checks+=1
        oracles={int(r['asset']):float(r['oracle_error_m']) for r in oldrows if r['variant']==v and r['method']=='material_patch'}
        gt_scores=[]
        for i in range(24):
            gt=np.array([poses[e,i]@inverse(poses[0,i]) for e in [1,2]]);gt_scores.append(score((i,*perm[1:,i]),gt))
        energy.append(dict(variant=v,methods=METHODS,true_global_energy=np.array(gt_scores).sum(0).tolist(),selected_global_energy=graph['values'].min(0).tolist()))
        for col,m in enumerate(METHODS):
            with np.load(root/f'{v}_{m}.npz') as f:pred=dict(f)
            h=int(pred['search_state']);assert h==int(graph['best'][col]) and np.array_equal(pred['target_slots'],graph['states'][h]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc)
            for i,(j,k) in enumerate(pred['target_slots']):
                j,k=int(j),int(k);T=pred['transforms'][i];q=lookup[i,j,k];assert np.array_equal(T,cache['transforms'][q,col]);val=score((i,j,k),T);error=abs(float(val[col])-float(pred['objectives'][i]));max_error=max(max_error,error);assert error<1e-8;assert val[2]>=val[0]-1e-12;checks+=3;gt=poses[2,i]@inverse(poses[0,i]);angle=float(np.degrees(Rotation.from_matrix(T[1,:3,:3]@gt[:3,:3].T).magnitude()));rr[i].update(rear=int(perm[2,i]%3==0),oracle_pool_qualified=int(oracles[i]<=.05),oracle_error_m=oracles[i],h2_rotation_error_deg=angle,true_pose_energy=float(gt_scores[i][col]),selected_energy=float(val[col]));assert rr[i]['qualified']<=rr[i]['oracle_pool_qualified'];checks+=1
                if col<2:
                    # Counterfactual changes only the normal factor at the SAME old pose/ID.
                    counter.append(dict(variant=v,asset=i,method=m,rear=rr[i]['rear'],both_ids=rr[i]['both_ids'],qualified=rr[i]['qualified'],raw_energy=float(val[col]),normal_energy=float(val[col+2]),normal_minus_raw=float(val[col+2]-val[col]),raw_true_energy=float(gt_scores[i][col]),normal_true_energy=float(gt_scores[i][col+2])))
                if m in ['raw_patch','normal_patch'] and rr[i]['both_ids']:
                    a=banks[0][i];world=np.einsum('ij,npj->npi',T[1,:3,:3],a['patch_xyz'])+T[1,None,None,:3,3];n=a['patch_normal']@T[1,:3,:3].T;raw=np.zeros(world.shape[:-1],bool);gated=np.zeros_like(raw);opposed=np.zeros_like(raw)
                    for f in views[2][k]:
                        r=project(world,f,k);hit=r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m']);cos=np.sum(n[:,None]*r['normal'],axis=-1);compatible=(cos>=cfg['normal_cosine_min'])|~a['patch_normal_valid'][:,None]|~r['normal_valid'];raw|=hit;gated|=hit&compatible;opposed|=hit&~compatible
                    normals.append(dict(variant=v,asset=i,method=m,rear=rr[i]['rear'],raw_hit_pixels=int(raw.sum()),normal_hit_pixels=int(gated.sum()),opposed_pixels=int(opposed.sum()),h2_rotation_error_deg=angle,worst_error_m=rr[i]['worst_error_m']))
            rows+=rr
            if col<2:
                prev=['projective_points','material_patch'][col]
                with np.load(parent/f'{v}_{prev}.npz') as f:pp=dict(f)
                assert pred['transforms'].shape==pp['transforms'].shape;checks+=1
        # Complete unpruned normal scoring on a public fixed triple.
        with np.load(parent/f'{v}_pools.npz') as f:pools=dict(f)
        i,j,k=map(int,cache['triples'][0]);A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]];T1=np.repeat(A,len(B),0);T2=np.tile(B,(len(A),1,1));sc=[]
        for pe,a,b in [(0,i,j),(1,i,k),(2,j,k)]:e,f=PAIRS[pe];sc.append(OrientedScore(banks[e][a],banks[f][b],views[e][a],views[f][b],a,b,cfg))
        vv=sc[0].score(A)[:,None]+sc[1].score(B)[None]+.5*sc[2].score(T2@inverse(T1)).reshape(len(A),len(B),4);minimum=vv.reshape(-1,4).min(0)
        with np.load(Path(cfg['coupled_parent_root'])/f'{v}_cache.npz') as f:cp=dict(f)
        match=np.flatnonzero(np.all(cp['triples']==[i,j,k],axis=1))
        if len(match):
            T=cp['transforms'][match[0]];minimum=np.minimum(minimum,(sc[0].score(T[:,0])+sc[1].score(T[:,1])+.5*sc[2].score(T[:,1]@inverse(T[:,0]))).min(0))
        error=float(np.max(abs(minimum-cache['costs'][lookup[i,j,k]])));assert error<1e-8;checks+=1;brute.append(dict(variant=v,pairs=len(T1),max_error=error))
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'same_pose_counterfactual.csv',counter);write_csv(root/'surface_sidedness.csv',normals);summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS:
            rr=[r for r in rows if r['variant']==v and r['method']==m];correct=[r for r in rr if r['both_ids']];rear=[r for r in correct if r['rear']];summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=len(correct),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),rear_correct_ids=len(rear),rear_over90_rotation=sum(r['h2_rotation_error_deg']>90 for r in rear),oracle_pool_qualified=sum(r['oracle_pool_qualified'] for r in rr)))
        for base,cand in [('raw_points','normal_points'),('raw_patch','normal_patch'),('normal_points','normal_patch')]:
            a={r['asset']:r for r in rows if r['variant']==v and r['method']==base};b={r['asset']:r for r in rows if r['variant']==v and r['method']==cand};common=[i for i in a if a[i]['both_ids'] and b[i]['both_ids']];comparisons.append(dict(variant=v,baseline=base,candidate=cand,gained=[i for i in a if b[i]['qualified'] and not a[i]['qualified']],lost=[i for i in a if a[i]['qualified'] and not b[i]['qualified']],common_correct_ids=len(common),paired_worst_error_change_median_m=float(np.median([b[i]['worst_error_m']-a[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(primary_rows=len(rows),summary=summary,comparisons=comparisons,true_pose_energy=energy,checks=checks,max_selected_objective_error=max_error,brute_force=brute,scope='Outcome-informed single TRAIN, fixed V9 candidate and identity graph. Slant-invariant inverse-plane patch/field correction, all four scores recomputed, no new poses/training; not novelty or generalization confirmation.'));print(json.dumps(dict(checks=checks,summary=summary,comparisons=comparisons),indent=2))

if __name__=='__main__':main()
