"""POST-SEAL oracle-ID ranking intervention, not new predictions/accuracy.

Hold data, kernels, priors and poses fixed. Change ONLY frozen cause proposal
to joint H+C1+C2 coordinate variational updates. Both geometries get same two
initializations/40-round budget. Uses GT only to select diagnostic poses/IDs.
"""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from inspection_decisions.surface_measure_first_return_v1 import (ray_data,local_data,surface_terms,voxel_ids,unknown_exposure,causes,gamma_integral)


def representation(d,T,slot,model,cfg):
    local=local_data(d,T,slot);K,As=surface_terms(local,model,cfg)
    return K,voxel_ids(local['P'],model,cfg),np.r_[unknown_exposure(d,T,cfg),As]


def optimize(K,ids,A,mass,aH,model,cfg):
    rate=np.ones(len(mass));b=rate+A;best=None
    for init in [aH,mass+np.r_[np.bincount(ids,minlength=model['V']),np.zeros(len(model['center']))]]:
        a=init.copy();trace=[]
        for iteration in range(40):
            n,c=causes(K,ids,a,b,model,cfg);a=mass+n;value=float(c+gamma_integral(mass,rate,n,A).sum());trace.append(value)
            if len(trace)>1:
                assert trace[-1]>=trace[-2]-1e-7
                if trace[-1]-trace[-2]<1e-6:break
        result=dict(ELBO=value,iterations=len(trace),last_improvement=trace[-1]-trace[-2] if len(trace)>1 else None)
        if best is None or value>best['ELBO']:best=result
    return best


def main():
    root=Path('results/surface_measure_first_return_v1');assert (root/'evaluation_sources.json').exists() and read(root/'audit.json')['passed'];rank=read(root/'postseal_ranking_diagnostic.json');lookup={(r['scene'],r['asset']):r for r in rank['rows'] if r['method']=='surface_shared'};rows=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=root/run['storage'];cfg=read(src/'scene_config.json');base=Path(cfg['base_root'])/tag
        with (base/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (base/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        frames,_,_=load_frames(Path(cfg['base_root']),tag,inp['T0'][:,:3,3],cfg);data=[ray_data(ff,cfg) for ff in frames];choices=np.load(src/'costs.npz')['choices'][0]
        for i in range(24):
            model=read(src/f'material_{i:02d}.json')['model']
            for key,shape in [('center',(-1,3)),('normal',(-1,3)),('precision',(-1,3,3)),('surface_mass',(-1,))]:model[key]=np.array(model[key]).reshape(shape)
            post=np.load(src/f'posterior_{i:02d}.npz');aH=post['a'];mass=np.r_[np.full(model['V'],cfg['gamma_shape']),model['surface_mass']]
            H=representation(data[0],inp['T0'][i],i,model,cfg);j,k=int(truth['slot1'][i]),int(truth['slot2'][i]);l,r=choices[i,j,k];sets={}
            for name,Ts in [('pool',[np.array(pools[i][0][l]['T']),np.array(pools[i][1][r]['T'])]),('true',truth['poses'][:,i])]:
                S=[H]+[representation(data[t+1],Ts[t],slot,model,cfg) for t,slot in enumerate([j,k])];K=np.concatenate([s[0] for s in S]);ids=np.concatenate([s[1] for s in S]);A=sum(s[2] for s in S);sets[name]=optimize(K,ids,A,mass,aH,model,cfg)
            old=lookup[(tag,i)];delta=sets['true']['ELBO']-sets['pool']['ELBO']
            rows.append(dict(scene=tag,history_views=run['history_views'],asset=i,frozen_pool_quant_wrong=not old['correct_ID_pool_quant_ok'],frozen_pool_minus_true=old['correct_ID_min_cost']-old['true_pose_cost'],joint_update_pool_minus_true=delta,wrong_ranking_reversed=bool(not old['correct_ID_pool_quant_ok'] and delta>1e-7),optimization=sets))
        print('SOURCE_INTERVENTION',tag,flush=True)
    summary=[]
    for h in [None,1,3]:
        rr=[r for r in rows if h is None or r['history_views']==h]
        summary.append(dict(history_views=h,n=len(rr),frozen_wrong=sum(r['frozen_pool_quant_wrong'] for r in rr),wrong_ranking_reversed=sum(r['wrong_ranking_reversed'] for r in rr)))
    save(root/'postseal_source_update_intervention.json',dict(POST_SEAL_GT_DIAGNOSTIC_ONLY=True,NOT_algorithm_success=True,source_sha256=sha(Path(__file__)),fixed_inference_sources_unchanged=True,rows=rows,summary=summary,scope='Ranking intervention at two fixed correct-ID geometries. Does not rerun global association, optimize poses, guarantee VB global convergence or compute exact evidence. Not counted as repaired tasks.'))
    print('SOURCE_INTERVENTION_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
