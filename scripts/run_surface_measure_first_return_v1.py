"""Full 192-task execution shards. No evaluator imports or truth file access."""
from pathlib import Path
import pickle,time,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from inspection_decisions.surface_measure_first_return_v1 import (ray_data,build_dictionary,history_posterior,unknown_exposure,candidate_stats,pair_costs,PoseObjective,refine,cost)
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from inspection_decisions.sparse_material_evidence_v1 import reduce_slots,assign
from scripts.run_first_return_gamma_material_v4 import load_frames
from scripts.run_first_return_gamma_material_v5 import repair
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    torch.set_num_threads(1);cp=Path('configs/surface_measure_first_return_v1.json');options=read(cp);previous=Path(options['previous_root']);geo=Path(options['geometry_root']);cfg={**read(previous/'config.json'),**options};shard=int(sys.argv[1]);assert 0<=shard<3
    out=Path(cfg['output_root'])/f'shard_{shard}';out.mkdir(parents=True,exist_ok=False)
    for p,h in read(previous/'prediction_seal.json')['sha256'].items():assert sha(previous/p)==h
    inputs=[previous/'prediction_seal.json',previous/'manifest.json',geo/'prediction_seal.json',Path('research/surface_measure_first_return_checks_v1.json'),Path('research/surface_measure_first_return_actual_probe_v1.json')]
    sources=[cp,Path(__file__),Path('inspection_decisions/surface_measure_first_return_v1.py'),Path('scripts/check_surface_measure_first_return_v1.py'),Path('scripts/probe_surface_measure_first_return_v1.py'),Path('research/surface_measure_first_return_contract_v1_2026-10-25.md')]
    for p in ['inspection_decisions/first_return_gamma_material_v1.py','inspection_decisions/sparse_material_evidence_v1.py','inspection_decisions/material_supported_continuous_alignment_v1.py','inspection_decisions/first_return_material_surface_v1.py','scripts/run_first_return_gamma_material_v4.py','scripts/run_first_return_gamma_material_v5.py','scripts/run_sparse_ray_material_incidence_v3.py']:
        sources.append(Path(p))
    parent=Path(cfg['parent_root']);base=Path(cfg['base_root']);limits=read(previous/'manifest.json')['motion_limits']
    inputs += [parent/'prediction_seal.json',parent/'manifest.json']+list(parent.glob('*/candidates.pkl'))+list(parent.glob('*/predictions.json'))+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'))
    for run in read(previous/'runs.json'):
        inputs += [geo/run['scene']/f'current_geometry_{t}.json' for t in range(2)]
        inputs += [geo/run['scene']/f'current_stats_{t}.npz' for t in range(2)]
    save(out/'config.json',cfg);save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},GT_read=False,motion_limits=limits,all_288_retained=True,pixel_exposure_quadrature=True,same_pool_not_equal_FLOPs=True))
    runs=[];begin=time.perf_counter()
    for ordinal,run in enumerate(read(previous/'runs.json')):
        if ordinal%3!=shard:continue
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter();local={**cfg,'hazard_extent_m':read(geo/tag/'scene_config.json')['hazard_extent_m']};save(dest/'scene_config.json',local)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (parent/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,colors,tokens=load_frames(base,tag,centers,local);data=[ray_data(ff,local) for ff in frames];save(dest/'raw_ray_inventory.json',dict(raw=tokens,quadrature=[dict(raw_pixels=d['raw_pixels'],quadrature_rays=d['quadrature_rays'],total_weight=float(d['weight'].sum()),positive_returns=len(d['ps'])) for d in data]))
        UA=[];maps=[]
        for t in range(2):
            descriptors=read(geo/tag/f'current_geometry_{t}.json')
            with np.load(geo/tag/f'current_stats_{t}.npz') as z:mapping=z['mapping'].copy()
            values=[]
            for k,b in enumerate(descriptors):
                values.append(unknown_exposure(data[t+1],np.array(b['T']),local))
                if k%400==399:print('GEOMETRY',tag,t,k+1,'s',round(time.perf_counter()-clock,1),flush=True)
            values=np.array(values);np.savez_compressed(dest/f'unknown_exposure_quad_{t}.npz',A=values,mapping=mapping);UA.append(values);maps.append(mapping)
            for i in range(24):
                assert len(pools[i][t])==288 and set(b['slot'] for b in pools[i][t])==set(range(24))
                for c,b in enumerate(pools[i][t]):
                    source=descriptors[mapping[i,c]];assert b['slot']==source['slot'] and np.array_equal(np.array(b['T']),source['T'])
        C=np.zeros((4,24,24,24));choices=np.zeros(C.shape+(2,),np.int16);appearance=np.zeros((24,2,288));unaries={m:np.zeros((2,24,24)) for m in cfg['methods']};models=[None]*24;posterior=[None]*24
        def asset(i):
            model=build_dictionary(frames[0],inp['T0'][i],i,local);a,b,meta=history_posterior(data[0],inp['T0'][i],i,model,local);models[i]=model;posterior[i]=(a,b);assert np.min(np.diff(meta['ELBO_trace']),initial=0)>-1e-6
            save(dest/f'material_{i:02d}.json',dict(model=model,history=meta));np.savez_compressed(dest/f'posterior_{i:02d}.npz',a=a,b=b)
            mu=colors[0][i].mean(0);rgb=np.array([[np.sum((colors[t+1][p['slot']]-mu)**2)/(2*local['appearance_sigma_rgb']**2) for p in pools[i][t]] for t in range(2)]);appearance[i]=rgb
            S=[]
            for t in range(2):
                cache={};ss=[]
                for ci,p in enumerate(pools[i][t]):
                    key=int(maps[t][i,ci])
                    if key not in cache:cache[key]=candidate_stats(data[t+1],np.array(p['T']),p['slot'],model,a,b,local,UA[t][key])
                    ss.append(cache[key])
                S.append(ss)
            matrices=pair_costs(a,b,*S,local['methods'])
            for mi,m in enumerate(local['methods']):
                matrix=matrices[m]+rgb[0,:,None]+rgb[1,None,:];C[mi,i],choices[mi,i]=reduce_slots(matrix,pools[i])
                for t in range(2):
                    # Same candidate unary proposal for the two integrated methods.
                    if m in ('surface_shared','surface_independent'):
                        from inspection_decisions.surface_measure_first_return_v1 import gamma_integral
                        u=np.array([-gamma_integral(a,b,s['n'],s['A']).sum()-s['c'] for s in S[t]])+rgb[t]
                    else:u=np.array([s['mean' if m=='surface_mean' else 'positive'] for s in S[t]])+rgb[t]
                    for slot in range(24):unaries[m][t,i,slot]=min(u[k] for k,p in enumerate(pools[i][t]) if p['slot']==slot)
                # Scalar rebuilding of every reduced ID pair, not only chosen ones.
                for j in range(24):
                    for k in range(24):
                        l,r=choices[mi,i,j,k];assert abs(cost(a,b,S[0][l],S[1][r],m)+rgb[0,l]+rgb[1,r]-C[mi,i,j,k])<1e-7
            print('ASSET',tag,i,'surfaces',len(model['center']),'s',round(time.perf_counter()-clock,1),flush=True)
        with ThreadPoolExecutor(max_workers=cfg['asset_cost_workers']) as pool:list(pool.map(asset,range(24)))
        np.savez_compressed(dest/'costs.npz',C=C,choices=choices,appearance=appearance)
        # Same start union for all methods: earlier fixed starts + all four unaries.
        starts=[tuple(np.array(x,dtype=int) for x in start) for start in read(geo/tag/'assignment_starts.json')]
        for m in local['methods']:starts.append(tuple(linear_sum_assignment(unaries[m][t])[1] for t in range(2)))
        save(dest/'assignment_starts.json',starts);assignments={}
        for mi,m in enumerate(local['methods']):
            a,b,log=assign(C[mi],starts,local['assignment_rounds']);previous_assignment=dict(slots=np.array([a,b]),log=log);assignments[m]=repair(C[mi],previous_assignment,local);print('ASSIGNED',tag,m,assignments[m]['log']['gap'],flush=True)
        save(dest/'assignments.json',assignments);rows=[]
        for i in range(24):
            row=dict(asset=i,methods={});a,b=posterior[i]
            for mi,m in enumerate(local['methods']):
                slots=assignments[m]['slots'][:,i];j,k=slots;l,r=choices[mi,i,j,k];p1,p2=pools[i][0][l],pools[i][1][r];x=np.r_[pose_vector(np.array(p1['T']),centers[j]),pose_vector(np.array(p2['T']),centers[k])]
                objective=PoseObjective(data[1:],slots,centers,models[i],a,b,local,m);initial=objective(x,False)
                assert abs(initial+appearance[i,0,l]+appearance[i,1,r]-C[mi,i,j,k])<1e-6,(m,initial,C[mi,i,j,k])
                p,log=refine(objective,x,limits,local)
                row['methods'][m]=dict(candidate_indices=[int(l),int(r)],initial_parameters=x,final_parameters=p,log=log,stages=dict(candidate=dict(slots=slots,T1=p1['T'],T2=p2['T']),final=dict(slots=slots,T1=world_pose(p[:6],centers[j]),T2=world_pose(p[6:],centers[k]))))
            rows.append(row)
            if i%4==3:save(dest/'predictions_in_progress.json',rows);print('REFINE',tag,i+1,'s',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',rows);runs.append({**run,'elapsed_s':time.perf_counter()-clock,'continuous_fits':96,'actual_methods':local['methods']});save(out/'runs.json',runs);print('SCENE_DONE',tag,'s',round(time.perf_counter()-clock,1),flush=True)
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h
    save(out/'prediction_seal.json',dict(complete=True,GT_read=False,subset_shard=shard,elapsed_s=time.perf_counter()-begin,sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}));print('SHARD_SEALED',shard,flush=True)


if __name__=='__main__':main()
