"""Feasibility of every accepted update, real measured features, exact pool use."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.continuous_material_pose_mainline_v1 import select
from inspection_decisions.material_supported_continuous_alignment_v1 import historical_offsets,pose_vector,material_residual


def main():
    root=Path('results/material_supported_continuous_alignment_v2');cfg=read(root/'config.json');m=read(root/'manifest.json');base=Path(cfg['base_root']);baseline=Path(cfg['baseline_root']);rmax,tmax=m['motion_limits'];checks=0;paths=states=0;rng=np.random.default_rng(77601);max_profile_error=0.
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in m[field].items():assert sha(name)==h;checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (dest/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (dest/'features.pkl').open('rb') as f:features=pickle.load(f)
        with (baseline/tag/'candidates.pkl').open('rb') as f:old=pickle.load(f)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;pred=read(dest/'predictions.json');costs=np.load(dest/'material_costs.npz')
        for i,bb in enumerate(candidates):
            reference=historical_offsets(features[0][i])
            for t,branches in enumerate(bb):
                for k,b in enumerate(branches):
                    assert np.isfinite(b['T']).all() and np.linalg.norm(Rotation.from_matrix(b['T'][:3,:3]).as_rotvec())<=rmax+1e-10;checks+=1
                    assert np.linalg.norm(b['T'][:3,3]-slots[b['slot']])<=tmax+1e-10;checks+=1
                    if k<54:
                        original=old[i][t][k];assert b['slot']==original['slot'] and b['source']=='catalogue' and np.array_equal(b['T'],original['T']) and b['scores']==original['scores'];checks+=1
            ind=costs[f'independent_{i}'];shared=costs[f'shared_{i}'];assert ind.shape==(len(bb[0]),len(bb[1])) and shared.shape==ind.shape and np.all(shared>=ind-1e-8);checks+=1
            # Check actual mixed-slot candidates, not only a single-slot toy.
            for _ in range(8):
                k=int(rng.integers(len(bb[0])));l=int(rng.integers(len(bb[1])));chosen=[bb[0][k],bb[1][l]];ff=[features[t+1][b['slot']] for t,b in enumerate(chosen)];x=np.concatenate([pose_vector(b['T'],slots[b['slot']]) for b in chosen])
                for coupled,matrix in [(False,ind),(True,shared)]:
                    r,_=material_residual(x,ff,reference,cfg,coupled);gap=abs(float(r@r)-matrix[k,l]);max_profile_error=max(max_profile_error,gap);assert gap<1e-7;checks+=1
        icp=read(dest/'bounded_icp_logs.json');expected={(i,t,k) for i in range(24) for t in range(2) for k in range(54)}
        assert len(icp)==2592 and {(l['asset'],l['epoch'],l['parent']) for l in icp}==expected;checks+=1;paths+=len(icp)
        for log in icp+read(dest/'joint_logs.json'):
            levels=log.get('levels',[log])
            for level in levels:
                for step in level['accepted_steps']:
                    parameters=np.array(step['parameters']).reshape(-1,6)
                    assert (np.linalg.norm(parameters[:,:3],axis=1)<=rmax+1e-10).all() and (np.linalg.norm(parameters[:,3:],axis=1)<=tmax+1e-10).all();checks+=1
                    assert step['after']<step['before'];checks+=1;states+=1
        for t in [0,1,2]:
            for v,path in enumerate(sorted((base/tag).glob(f'frame_{t}_*.npz'))):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                lab,world=labels(frame,slots,groups,cfg)
                for slot in range(24):
                    points=world[lab==slot]-slots[slot]
                    for f in [f for f in features[t][slot] if f['view']==v]:
                        assert f['validity']['resolved'] and f['validity']['second_singular_value_m']>f['validity']['noise_frobenius_bound_m'] and f['validity']['camera_plane_distance_m']>f['validity']['point_noise_bound_m'];checks+=1
                        measured=points[f['measured_indices']];assert len(measured)==f['count'] and len(measured)>=cfg['plane_min_points'];checks+=1
                        assert np.allclose(measured.mean(0),f['centroid'],atol=1e-12);checks+=1
                        assert f['normal']@(frame['C'][:3,3]-slots[slot]-f['centroid'])>=-1e-10;checks+=1
        for method,result in pred.items():
            assert result['status']==0 and len(result['selection'])==24;checks+=1
            for t in [0,1]:assert len({candidates[i][t][pair[t]]['slot'] for i,*pair in result['selection']})==24;checks+=1
            if method.startswith('material_') or method.startswith('catalogue_material_'):
                which='shared' if method.endswith('shared') else 'independent';total=sum(float(costs[f'{which}_{i}'][k,l]) for i,k,l in result['selection']);assert abs(total-result['objective'])<1e-7;checks+=1
                if method.startswith('catalogue_'):assert all(k<54 and l<54 for i,k,l in result['selection']);checks+=1
            if method.startswith('common_'):
                assert result==select(candidates,method[len('common_'):],True);checks+=1
    cache=Path(cfg['cached_icp_root'])
    for run in read(root/'runs.json'):
        tag=run['scene'];assert read(root/tag/'bounded_icp_logs.json')==read(cache/tag/'bounded_icp_logs.json');checks+=1
        old_pred=read(cache/tag/'predictions.json');new_pred=read(root/tag/'predictions.json')
        for method in ['bounded_icp_nn','bounded_icp_plane']:assert new_pred[method]==old_pred[method];checks+=1
    assert m['ICP_reexecuted'] is False and paths==20736;checks+=1
    rows=read(root/'scores.json');assert len(rows)==192*8 and all(r['resolved'] and r['status']==0 for r in rows);checks+=1
    result={'status':'PASS','checks':checks,'all_initial_paths':paths,'accepted_states_checked':states,'max_actual_profile_vs_statistics_error':max_profile_error,
        'analysis_sources':{str(p):sha(p) for p in [Path(__file__),Path('scripts/score_material_supported_continuous_alignment_v2.py'),Path('scripts/diagnose_material_supported_continuous_alignment_v2.py'),Path('scripts/check_material_support_depth_gauge_v2.py')]}}
    save(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
