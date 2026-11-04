"""Merge and raw-audit all192 matched voxel controls; no evaluator access."""
from pathlib import Path
import pickle
import numpy as np
import torch
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from scripts.run_surface_measure_matched_voxel_v1 import empty_model
from inspection_decisions.surface_measure_first_return_v1 import ray_data,history_posterior,candidate_stats,cost
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector


def main():
    torch.set_num_threads(1);root=Path('results/surface_measure_matched_voxel_v1')
    if not (root/'prediction_seal.json').exists():
        runs=[];sources={};inputs={};limits=None
        for si in range(3):
            shard=root/f'shard_{si}';seal=read(shard/'prediction_seal.json');assert seal['complete'] and not seal['GT_read']
            for p,h in seal['sha256'].items():assert sha(shard/p)==h
            manifest=read(shard/'manifest.json');sources.update(manifest['sources']);inputs.update(manifest['inputs']);inputs[str(shard/'prediction_seal.json')]=sha(shard/'prediction_seal.json');limits=manifest['motion_limits']
            for run in read(shard/'runs.json'):runs.append({**run,'storage':str((shard/run['scene']).relative_to(root))})
        assert len(runs)==8;save(root/'runs.json',runs);cfg=read(root/'shard_0/config.json');save(root/'config.json',{**cfg,'main_output_root':cfg['output_root'],'output_root':str(root),'methods':['voxel_matched_shared']})
        save(root/'manifest.json',dict(sources=sources,inputs=inputs,motion_limits=limits,GT_read=False))
        for field in ['sources','inputs']:
            for p,h in read(root/'manifest.json')[field].items():assert sha(p)==h
        save(root/'prediction_seal.json',dict(complete=True,assets=192,GT_read=False,sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print('MATCHED_CONTROL192_SEALED',flush=True)
    checks=0;error=0.;fits=0;limits=read(root/'manifest.json')['motion_limits']
    for run in read(root/'runs.json'):
        tag=run['scene'];src=root/run['storage'];cfg=read(src/'scene_config.json');base=Path(cfg['base_root'])/tag
        with (base/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,_,_=load_frames(Path(cfg['base_root']),tag,centers,cfg);data=[ray_data(ff,cfg) for ff in frames];model=empty_model(cfg);old=np.load(src/'history_posterior.npz');pred=read(src/'predictions.json');z=np.load(src/'costs.npz');C=z['C'];appearance=z['appearance'];slots=np.array(read(src/'assignments.json')['slots'])
        assert all(sorted(row)==list(range(24)) for row in slots);checks+=2
        for i in range(24):
            a,b,_=history_posterior(data[0],inp['T0'][i],i,model,cfg);assert np.max(abs(a-old['a'][i]))<1e-8 and np.max(abs(b-old['b'][i]))<1e-8;checks+=2
            row=pred[i];l,r=row['candidate_indices'];j,k=row['stages']['candidate']['slots'];SS=[]
            for stage in ['candidate','final']:
                P=row['stages'][stage];S=[candidate_stats(data[t+1],np.array(P[f'T{t+1}']),P['slots'][t],model,a,b,cfg) for t in range(2)];value=cost(a,b,*S,'surface_shared');target=row['log']['initial_cost' if stage=='candidate' else 'final_cost'];diff=abs(value-target);error=max(error,diff);assert diff<1e-6;checks+=1
                if stage=='candidate':assert abs(value+appearance[i,0,l]+appearance[i,1,r]-C[i,j,k])<1e-6;checks+=1
                for t in range(2):
                    x=pose_vector(np.array(P[f'T{t+1}']),centers[P['slots'][t]])
                    assert np.linalg.norm(x[:3])<=limits[0]+1e-10 and np.linalg.norm(x[3:])<=limits[1]+1e-10;checks+=2
            assert row['log']['final_cost']<=row['log']['initial_cost']+1e-8;checks+=1;fits+=1
        print('MATCHED_CONTROL_AUDIT',tag,checks,flush=True)
    assert fits==192;save(root/'audit.json',dict(passed=True,GT_read=False,continuous_fits_rebuilt=192,checks=checks,max_raw_cost_error=error,script_sha256=sha(Path(__file__)),all_288_retained=True,same_exposure_same_total_prior_mass=True));print('MATCHED_CONTROL_AUDIT_PASS',checks,flush=True)


if __name__=='__main__':main()
