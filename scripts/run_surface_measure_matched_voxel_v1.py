"""Same-background-quadrature pure voxel control, all192 before any new GT."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle,time,sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from scripts.run_first_return_gamma_material_v5 import repair
from inspection_decisions.surface_measure_first_return_v1 import ray_data,unknown_exposure,PoseObjective,refine
from inspection_decisions.first_return_gamma_fast_v1 import pair_costs
from inspection_decisions.sparse_material_evidence_v1 import reduce_slots,assign
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose


def empty_model(cfg):
    q=int(round(2*cfg['hazard_extent_m']/cfg['hazard_voxel_m']))
    return dict(V=q**3,q=q,center=np.empty((0,3)),normal=np.empty((0,3)),precision=np.empty((0,3,3)),footprint=np.empty(0),surface_mass=np.empty(0))


def main():
    torch.set_num_threads(1);mainroot=Path('results/surface_measure_first_return_v1');assert not (mainroot/'evaluation_sources.json').exists();cp=Path('configs/surface_measure_first_return_v1.json');cfg={**read('results/first_return_gamma_material_v5/config.json'),**read(cp)};shard=int(sys.argv[1]);out=Path('results/surface_measure_matched_voxel_v1')/f'shard_{shard}';out.mkdir(parents=True,exist_ok=False)
    base=Path(cfg['base_root']);parent=Path(cfg['parent_root']);geo=Path(cfg['geometry_root']);limits=read('results/first_return_gamma_material_v5/manifest.json')['motion_limits']
    sources=[Path(__file__),cp,Path('inspection_decisions/surface_measure_first_return_v1.py'),Path('inspection_decisions/first_return_gamma_fast_v1.py'),Path('inspection_decisions/first_return_gamma_material_v1.py'),Path('inspection_decisions/sparse_material_evidence_v1.py'),Path('scripts/run_first_return_gamma_material_v4.py'),Path('scripts/run_first_return_gamma_material_v5.py'),Path('research/surface_measure_matched_voxel_control_v1_2026-10-25.md')]
    manifest=dict(sources={str(p):sha(p) for p in sources},inputs={str(geo/'prediction_seal.json'):sha(geo/'prediction_seal.json')},GT_read=False,motion_limits=limits,same_background_quadrature=True,surface_mass_redistributed_to_voxels=True);save(out/'manifest.json',manifest);save(out/'config.json',cfg);runs=[]
    for ordinal,run in enumerate(read('results/first_return_gamma_material_v5/runs.json')):
        if ordinal%3!=shard:continue
        tag=run['scene'];src=mainroot/f'shard_{shard}'/tag
        while not (src/'material_00.json').exists():time.sleep(5)
        local=read(src/'scene_config.json');model=empty_model(local);V=model['V'];pure={**local,'gamma_shape':local['gamma_shape']+local['surface_prior_mass']/V}
        dest=out/tag;dest.mkdir();save(dest/'scene_config.json',local);clock=time.perf_counter()
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (parent/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,colors,_=load_frames(base,tag,centers,local);data=[ray_data(ff,local) for ff in frames]
        with np.load(geo/tag/'history_stats.npz') as z:H=z['H'].copy()
        H[:,1]=np.array([unknown_exposure(data[0],inp['T0'][i],local)*local['hazard_voxel_m'] for i in range(24)]);H[:,2]=0
        counts=[];exposures=[];maps=[]
        used=[src/'scene_config.json',geo/tag/'history_stats.npz',base/tag/'inference.pkl',parent/tag/'candidates.pkl']+list((base/tag).glob('frame_*.npz'))
        for t in range(2):
            with np.load(geo/tag/f'current_stats_{t}.npz') as z:counts.append(z['Q'][:,0,:].copy());mapping=z['mapping'].copy()
            with np.load(src/f'unknown_exposure_quad_{t}.npz') as z:exposures.append(z['A'].copy());assert np.array_equal(mapping,z['mapping'])
            maps.append(mapping);used += [geo/tag/f'current_stats_{t}.npz',src/f'unknown_exposure_quad_{t}.npz']
        C=np.zeros((24,24,24));choices=np.zeros(C.shape+(2,),np.int16);appearance=np.zeros((24,2,288));posterior=[]
        for i in range(24):posterior.append((pure['gamma_shape']+H[i,0],(pure['gamma_rate_m']+H[i,1])/local['hazard_voxel_m']))
        np.savez_compressed(dest/'history_posterior.npz',a=np.array([p[0] for p in posterior]),b=np.array([p[1] for p in posterior]))
        def asset(i):
            Q=[np.stack([counts[t][maps[t][i]],exposures[t][maps[t][i]]*local['hazard_voxel_m'],np.zeros_like(counts[t][maps[t][i]])],axis=1) for t in range(2)]
            mu=colors[0][i].mean(0);rgb=np.array([[np.sum((colors[t+1][p['slot']]-mu)**2)/(2*local['appearance_sigma_rgb']**2) for p in pools[i][t]] for t in range(2)]);appearance[i]=rgb
            matrix=pair_costs(H[i],*Q,pure,'gamma_joint')+rgb[0,:,None]+rgb[1,None,:];C[i],choices[i]=reduce_slots(matrix,pools[i])
            print('CONTROL_COST',tag,i,'s',round(time.perf_counter()-clock,1),flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(asset,range(24)))
        while not (src/'assignment_starts.json').exists():time.sleep(5)
        starts=[tuple(np.array(x,int) for x in s) for s in read(src/'assignment_starts.json')];used.append(src/'assignment_starts.json')
        j,k,log=assign(C,starts,local['assignment_rounds']);assignment=repair(C,dict(slots=np.array([j,k]),log=log),local);save(dest/'assignments.json',assignment);np.savez_compressed(dest/'costs.npz',C=C,choices=choices,appearance=appearance);predictions=[]
        for i in range(24):
            slots=assignment['slots'][:,i];j,k=slots;l,r=choices[i,j,k];p1,p2=pools[i][0][l],pools[i][1][r];x=np.r_[pose_vector(np.array(p1['T']),centers[j]),pose_vector(np.array(p2['T']),centers[k])];a,b=posterior[i]
            objective=PoseObjective(data[1:],slots,centers,model,a,b,local,'surface_shared');initial=objective(x,False);assert abs(initial+appearance[i,0,l]+appearance[i,1,r]-C[i,j,k])<1e-6
            p,fit=refine(objective,x,limits,local);predictions.append(dict(asset=i,candidate_indices=[int(l),int(r)],log=fit,stages=dict(candidate=dict(slots=slots,T1=p1['T'],T2=p2['T']),final=dict(slots=slots,T1=world_pose(p[:6],centers[j]),T2=world_pose(p[6:],centers[k])))))
        assert not (mainroot/'evaluation_sources.json').exists()
        save(dest/'predictions.json',predictions);runs.append({**run,'continuous_fits':24,'elapsed_s':time.perf_counter()-clock});save(out/'runs.json',runs)
        manifest['inputs'].update({str(p):sha(p) for p in used});save(out/'manifest.json',manifest);print('CONTROL_DONE',tag,'s',round(time.perf_counter()-clock,1),flush=True)
    for field in ['sources','inputs']:
        for p,h in manifest[field].items():assert sha(p)==h
    save(out/'prediction_seal.json',dict(complete=True,GT_read=False,sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}));print('CONTROL_SHARD_SEALED',shard,flush=True)


if __name__=='__main__':main()
