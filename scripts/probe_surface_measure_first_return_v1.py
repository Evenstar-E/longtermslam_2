"""Runtime / raw-exposure discrepancy probes only, no evaluator access."""
import pickle,time
from pathlib import Path
import numpy as np
import torch
from scripts.run_rigid_identity_bounds_v1 import read
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.surface_measure_first_return_v1 import *


def main():
    torch.set_num_threads(1);start=time.perf_counter();cfg={**read('results/first_return_gamma_material_v5/config.json'),**read('configs/surface_measure_first_return_v1.json')};rows=[]
    for tag in ['aligned_s270100_h1','aligned_s270100_h3']:
        cfg={**cfg,**{'hazard_extent_m':read(Path(cfg['geometry_root'])/tag/'scene_config.json')['hazard_extent_m']}}
        base=Path(cfg['base_root'])
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,_,_=load_frames(base,tag,centers,cfg);data=[ray_data(ff,cfg) for ff in frames]
        model=build_dictionary(frames[0],inp['T0'][0],0,cfg);a,b,meta=history_posterior(data[0],inp['T0'][0],0,model,cfg)
        assert np.min(np.diff(meta['ELBO_trace']))>-1e-6
        probe=pools[0][0][0];T=np.array(probe['T']);slot=probe['slot'];clock=time.perf_counter();s=candidate_stats(data[1],T,slot,model,a,b,cfg);elapsed=time.perf_counter()-clock
        raw=ray_data(frames[1],{**cfg,'exposure_stride':1});_,quadA=surface_terms(local_data(data[1],T,slot),model,cfg);_,rawA=surface_terms(local_data(raw,T,slot),model,cfg)
        absolute=float(np.linalg.norm(quadA-rawA));relative=absolute/max(float(np.linalg.norm(rawA)),1e-12)
        p2=pools[0][1][0];x=np.r_[pose_vector(T,centers[slot]),pose_vector(np.array(p2['T']),centers[p2['slot']])]
        objective=PoseObjective(data[1:],[slot,p2['slot']],centers,model,a,b,cfg,'surface_shared');v,g=objective(x)
        s2=candidate_stats(data[2],np.array(p2['T']),p2['slot'],model,a,b,cfg)
        error=abs(v-cost(a,b,s,s2,'surface_shared'));assert error<1e-7
        row=dict(scene=tag,asset=0,history=meta,candidate_seconds=elapsed,quadrature_surface_exposure_relative_error=relative,
                 quadrature_surface_exposure_absolute_error=absolute,raw_surface_exposures=rawA,quad_surface_exposures=quadA,
                 numpy_torch_actual_error=error,pose_gradient_norm=float(np.linalg.norm(g)),finite_gradient=bool(np.isfinite(g).all()),GT_read=False)
        assert row['finite_gradient'];rows.append(row);print('PROBE',tag,'surfaces',len(model['center']),'candidate_s',elapsed,'quad_rel',relative,'backend_error',error,flush=True)
    save('research/surface_measure_first_return_actual_probe_v1.json',dict(passed=True,rows=rows,elapsed_s=time.perf_counter()-start,GT_read=False,not_task_scores=True))


if __name__=='__main__':main()
