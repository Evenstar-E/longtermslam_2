"""POST-SEAL full192 correct-ID pool ranking; never an algorithm success rate."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box,transform
from inspection_decisions.surface_measure_first_return_v1 import ray_data,candidate_stats,cost
from scripts.run_surface_measure_matched_voxel_v1 import empty_model


def main():
    root=Path('results/surface_measure_first_return_v1');assert read(root/'prediction_seal.json')['complete'] and read(root/'audit.json')['passed'] and (root/'evaluation_sources.json').exists()
    rows=[];evidence=read(root/'evaluation_sources.json');actual={(r['scene'],r['asset'],r['method']):r for r in read(root/'scores.json') if r['family']=='surface' and r['stage']=='final'}
    matched_actual={(r['scene'],r['asset']):r for r in read(root/'scores.json') if r['family']=='matched_voxel' and r['stage']=='final'};controlroot=Path('results/surface_measure_matched_voxel_v1')
    for run in read(root/'runs.json'):
        tag=run['scene'];src=root/run['storage'];cfg=read(src/'scene_config.json');base=Path(cfg['base_root'])/tag
        assert sha(base/'evaluation_only.pkl')==evidence['truth_sha256'][tag]
        with (base/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (base/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        frames,_,_=load_frames(Path(cfg['base_root']),tag,inp['T0'][:,:3,3],cfg);data=[ray_data(ff,cfg) for ff in frames[1:]];zz=np.load(src/'costs.npz');C=zz['C'];choices=zz['choices'];appearance=zz['appearance'];geometry=geometry_for(run['seed'],24,run['geometry'])
        cr=next(r for r in read(controlroot/'runs.json') if r['scene']==tag);cs=controlroot/cr['storage'];cz=np.load(cs/'costs.npz');cp=np.load(cs/'history_posterior.npz');assert np.max(abs(cz['appearance']-appearance))<1e-8
        for i in range(24):
            model=read(src/f'material_{i:02d}.json')['model']
            for key,shape in [('center',(-1,3)),('normal',(-1,3)),('precision',(-1,3,3)),('footprint',(-1,)),('surface_mass',(-1,))]:model[key]=np.array(model[key],float).reshape(shape)
            posterior=np.load(src/f'posterior_{i:02d}.npz');a,b=posterior['a'],posterior['b'];j,k=int(truth['slot1'][i]),int(truth['slot2'][i]);trueT=truth['poses'][:,i];true_indices=[]
            for t,slot in enumerate([j,k]):
                distance=np.array([np.max(abs(np.array(p['T'])-trueT[t])) if p['slot']==slot else np.inf for p in pools[i][t]]);index=int(distance.argmin());assert distance[index]<1e-8;true_indices.append(index)
            trueS=[candidate_stats(data[t],trueT[t],slot,model,a,b,cfg) for t,slot in enumerate([j,k])]
            ray=inp['historical_rays'][i];o=np.array(ray['origin']);u=np.array(ray['ray']);s=min(float(ray_box(o,u,lo,hi)) for lo,hi in geometry[i]);material=o+s*u;anchor=inp['anchors'][i]
            truthp=np.array([transform(material[None],T)[0] for T in trueT]);true_delta=truthp[1]-truthp[0]
            for mi,m in enumerate(cfg['methods']):
                l,r=choices[mi,i,j,k];P=[np.array(pools[i][0][l]['T']),np.array(pools[i][1][r]['T'])];p=np.array([transform(anchor[None],T)[0] for T in P]);error=float(np.linalg.norm(p[1]-p[0]-true_delta));true_value=cost(a,b,*trueS,m)+appearance[i,0,true_indices[0]]+appearance[i,1,true_indices[1]];minimum=float(C[mi,i,j,k]);assert minimum<=true_value+1e-6
                q=actual[(tag,i,m)]
                rows.append(dict(scene=tag,history_views=run['history_views'],asset=i,method=m,true_pose_retained=True,correct_ID_pool_error_m=error,correct_ID_pool_quant_ok=bool(error<=.05),correct_ID_min_cost=minimum,true_pose_cost=true_value,strict_wrong_ranking=bool(error>.05 and minimum<true_value-1e-7),actual_success=q['success'],actual_both_ids=q['both_ids'],history_surface_hit_fraction=float((a[model['V']:]-model['surface_mass']).sum()/max(read(src/f'material_{i:02d}.json')['history']['history_hits'],1))))
            vm=empty_model(cfg);va,vb=cp['a'][i],cp['b'][i];S=[candidate_stats(data[t],trueT[t],slot,vm,va,vb,cfg) for t,slot in enumerate([j,k])];l,r=cz['choices'][i,j,k];P=[np.array(pools[i][0][l]['T']),np.array(pools[i][1][r]['T'])];pp=np.array([transform(anchor[None],T)[0] for T in P]);error=float(np.linalg.norm(pp[1]-pp[0]-true_delta));true_value=cost(va,vb,*S,'surface_shared')+appearance[i,0,true_indices[0]]+appearance[i,1,true_indices[1]];minimum=float(cz['C'][i,j,k]);assert minimum<=true_value+1e-6;q=matched_actual[(tag,i)]
            rows.append(dict(scene=tag,history_views=run['history_views'],asset=i,method='matched_voxel_shared',true_pose_retained=True,correct_ID_pool_error_m=error,correct_ID_pool_quant_ok=bool(error<=.05),correct_ID_min_cost=minimum,true_pose_cost=true_value,strict_wrong_ranking=bool(error>.05 and minimum<true_value-1e-7),actual_success=q['success'],actual_both_ids=q['both_ids'],history_surface_hit_fraction=0.))
        print('DIAGNOSED',tag,flush=True)
    summary=[]
    for m in cfg['methods']+['matched_voxel_shared']:
        for h in [None,1,3]:
            rr=[r for r in rows if r['method']==m and (h is None or r['history_views']==h)]
            summary.append(dict(method=m,history_views=h,n=len(rr),correct_ID_pool_quant_ok=sum(r['correct_ID_pool_quant_ok'] for r in rr),correct_ID_pool_wrong=sum(not r['correct_ID_pool_quant_ok'] for r in rr),strict_wrong_ranking=sum(r['strict_wrong_ranking'] for r in rr),actual_wrong_ID=sum(not r['actual_both_ids'] for r in rr),actual_correct_ID_quant_fail=sum(r['actual_both_ids'] and not r['actual_success'] for r in rr)))
    save(root/'postseal_ranking_diagnostic.json',dict(POST_SEAL_GT_DIAGNOSTIC_ONLY=True,NOT_algorithm_success=True,all_true_poses_retained=True,rows=rows,summary=summary,script_sha256=sha(Path(__file__)),conclusion_scope='If retained true-pose candidate loses to a wrong correct-ID pool optimum, candidate absence or finite identity-assignment search cannot fully explain that error. Does not identify footprint, noise, priors or variational approximation as sole cause.'))
    print('RANKING_DIAGNOSTIC_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
