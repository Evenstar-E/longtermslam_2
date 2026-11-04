"""Private labels AFTER prediction seal: tasks, pose oracle and material links."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv
from pathlib import Path
import numpy as np
from scripts.run_material_surface_correspondence_v13 import configuration,METHODS,load,AllScore,setup,PAIRS
from scripts.cache_material_surface_geometry_v9 import unpack
from inspection_decisions.material_surface_correspondence_v13 import ChartPair
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply,world_points
from scripts.score_material_native_modes_v6 import physical_error
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def private_chart_material(scene,charts,ids):
    records=json.loads((scene/'frames.json').read_text(encoding='utf-8')))
    for e,row in enumerate(charts):
        for a in row:a['q']=np.zeros_like(a['xyz']);a['physical_id']=np.full(len(a['xyz']),-1,int)
        for fi in np.unique(np.concatenate([a['frame'] for a in row])):
            rec=records[int(ids[e][int(fi)])]
            with np.load(scene/rec['labels']) as f:q=f['material_xyz'];physical=f['instance']
            for a in row:
                h=np.flatnonzero(a['frame']==fi);uv=a['uv'][h];a['q'][h]=q[uv[:,1],uv[:,0]];a['physical_id'][h]=physical[uv[:,1],uv[:,0]]


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene)
    rows=[];oracles=[];links=[];mechanism=[];maxerror=0.;paths=0;provenance=0
    for v in cfg['variants']:
        _,frames,ids,bc,_,_,views,banks,_=setup(cfg,v);charts=unpack(Path(cfg['chart_cache'])/f'{v}_charts.npz');old=load(Path(cfg['previous_root'])/f'{v}_cache.npz');pg=load(Path(cfg['previous_root'])/f'{v}_search_graph.npz');cache=load(root/f'{v}_cache.npz');graph=load(root/f'{v}_search_graph.npz');proposals=load(root/f'{v}_proposals.npz');fits=load(root/f'{v}_fits.npz');previous_proposals=load(Path(cfg['previous_root'])/f'{v}_proposals.npz')
        assert np.array_equal(cache['triples'],old['triples']) and np.array_equal(graph['states'],pg['states']);assert np.max(abs(cache['costs'][:,0]-old['costs'][:,6]))<1e-12;checks+=2
        lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
        for key,path in fits.items():
            if key.endswith('_path'):assert np.max(np.diff(path),initial=0)<=1e-12;checks+=1;paths+=1
        for state,value in zip(graph['states'],graph['values']):
            assert all(np.array_equal(np.sort(state[:,e]),np.arange(24)) for e in range(2));rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(8));assert np.max(abs(rebuilt-value))<1e-8;checks+=2
        for e,row in enumerate(charts):
            for slot,a in enumerate(row):
                h=a['center_index'];assert np.array_equal(a['xyz'],banks[e][slot]['xyz'][h]) and np.array_equal(a['uv'],banks[e][slot]['uv'][h]);checks+=1;provenance+=len(h)
                for fi in np.unique(a['frame']):
                    ix=a['frame']==fi;assert np.max(abs(world_points(frames[e][int(fi)],a['uv'][ix])-a['xyz'][ix]))<1e-12;checks+=1
                B=a['basis'];assert np.max(abs(B.swapaxes(-1,-2)@B-np.eye(3)))<1e-8 and np.min(np.linalg.det(B))>.999999;checks+=1
        for q in range(len(cache['triples'])):
            U=proposals[f'{q}_transforms'];R=U[:,:,:3,:3];assert np.isfinite(U).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
            for T in list(previous_proposals[f'{q}_transforms'])+list(old['transforms'][q]):assert any(np.max(abs(T-Q))<1e-9 for Q in U);checks+=1
            rebuilt=proposals[f'{q}_values'].min(0);assert np.max(abs(rebuilt-cache['costs'][q,1:]))<1e-8;checks+=1
        def scorer(i,j,k):return AllScore([ChartPair(charts[e][a],charts[f][b],views[e][a],views[f][b],a,b,cfg) for (e,f),a,b in zip(PAIRS,[i,i,j],[j,k,k])],cfg)
        private_chart_material(scene,charts,ids);oracle_by_i={}
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);errors={};h=lookup.get((i,j,k))
            if h is not None:
                U=proposals[f'{h}_transforms'];stage=proposals[f'{h}_stage'];err=np.array([physical_error(*T,x,y1,y2) for T in U])
                for name,lim in [('retained',0),('global',1),('continuous',2)]:errors[name]=float(err[stage<=lim].min())
                best=int(err.argmin());GT=np.array([poses[e,i]@inverse(poses[0,i]) for e in [1,2]]);truevalues=scorer(i,j,k).score(GT[None])[0];bestvalues=proposals[f'{h}_values'][best]
                mechanism.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),best_error_m=float(err[best]),best_stage=int(stage[best]),true_phase=float(truevalues[5]),true_shared_phase=float(truevalues[6]),best_phase=float(bestvalues[5]),best_shared_phase=float(bestvalues[6])))
            else:errors={name:None for name in ['retained','global','continuous']}
            oracle_by_i[i]=errors;oracles.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),correct_identity_searched=int(h is not None),**{name+'_error_m':val for name,val in errors.items()}))
        for col,m in enumerate(METHODS):
            p=load(root/f'{v}_{m}.npz');s=int(p['search_state']);assert s==int(graph['best'][col]) and np.array_equal(p['target_slots'],graph['states'][s]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,p,truth,anchors,bc)
            for i,(j,k) in enumerate(p['target_slots']):
                j,k=int(j),int(k);h=lookup[i,j,k];T=p['transforms'][i];assert np.array_equal(T,cache['transforms'][h,col]);checks+=1;sc=scorer(i,j,k)
                if col:
                    value=sc.score(T[None])[0,col-1];error=abs(float(value)-float(p['objectives'][i]));assert error<1e-8;maxerror=max(maxerror,error);checks+=1
                    if col>=2:
                        corecol=col-2;vv,evs,states=sc.core.evaluate(T,True);ev,state=evs[corecol],states[corecol]
                        for pe,((e,f),pair) in enumerate(zip(PAIRS,sc.pairs)):
                            if corecol%2:
                                cl=state['shared'];ok=(cl[:,e]>=0)&(cl[:,f]>=0);matched=cl[ok][:,[e,f]]
                            else:matched=ev[pe]['matched']
                            ia,ib=matched.T if len(matched) else (np.zeros(0,int),np.zeros(0,int));a,b=pair.a,pair.b;correct=(a['physical_id'][ia]==b['physical_id'][ib])&(np.linalg.norm(a['q'][ia]-b['q'][ib],axis=1)<=.015)
                            links.append(dict(variant=v,method=m,asset=i,pair=pe,rear=int(perm[2,i]%3==0),links=len(matched),correct_links=int(correct.sum()),material_radius_m=.015))
                o=oracle_by_i[i]['retained' if col==0 else 'continuous'];rr[i].update(rear=int(perm[2,i]%3==0),oracle_error_m=o,oracle_qualified=int(o is not None and o<=.05));assert rr[i]['qualified']<=rr[i]['oracle_qualified'];checks+=1
            rows+=rr
        p=load(Path(cfg['previous_root'])/f'{v}_independent_material.npz');q=load(root/f'{v}_retained_v12.npz');assert np.array_equal(p['target_slots'],q['target_slots']) and np.array_equal(p['transforms'],q['transforms']);checks+=1
        # Recompute EVERY endpoint, without using cached values, on a fixed public triple.
        i,j,k=map(int,cache['triples'][0]);values=scorer(i,j,k).score(proposals['0_transforms']);assert np.max(abs(values-proposals['0_values']))<1e-8;checks+=1
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_by_stage.csv',oracles);write_csv(root/'material_links.csv',links);write_csv(root/'pose_mechanism.csv',mechanism);summary=[];comparisons=[]
    for v in cfg['variants']:
        for m in METHODS:
            rr=[r for r in rows if r['variant']==v and r['method']==m];correct=[r for r in rr if r['both_ids']];ll=[r for r in links if r['variant']==v and r['method']==m];n=sum(r['links'] for r in ll);c=sum(r['correct_links'] for r in ll)
            summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=len(correct),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_qualified=sum(r['oracle_qualified'] for r in rr),correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in correct])) if correct else None,material_links=n,correct_material_links=c,link_precision=c/n if n else None))
        for a,b in [('retained_v12','multiscale_fusion'),('multiscale_fusion','partial_points'),('partial_points','partial_normals'),('partial_normals','partial_phase'),('partial_points','shared_points'),('partial_normals','shared_normals'),('partial_phase','shared_phase')]:
            aa={r['asset']:r for r in rows if r['variant']==v and r['method']==a};bb={r['asset']:r for r in rows if r['variant']==v and r['method']==b};common=[i for i in aa if aa[i]['both_ids'] and bb[i]['both_ids']]
            comparisons.append(dict(variant=v,baseline=a,candidate=b,gained=[i for i in aa if bb[i]['qualified'] and not aa[i]['qualified']],lost=[i for i in aa if aa[i]['qualified'] and not bb[i]['qualified']],common_correct_ids=len(common),paired_error_change_median_m=float(np.median([bb[i]['worst_error_m']-aa[i]['worst_error_m'] for i in common])) if common else None))
    save(root/'scores.json',dict(primary_rows=len(rows),checks=checks,fit_paths=paths,measured_centers_verified=provenance,max_selected_objective_error=maxerror,summary=summary,comparisons=comparisons,scope='Same-feature finite common coupled graph, not exhaustiveV12Cartesian replay. One seenTRAIN, all24x4, stronger observed metrictexture representation changes queries. Point/normal/phase and independent/shared controls isolate mechanisms. Localframes/correlation/assignment/GN known tools, originality and independent generalization unconfirmed.'));print(json.dumps(dict(checks=checks,summary=summary,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
