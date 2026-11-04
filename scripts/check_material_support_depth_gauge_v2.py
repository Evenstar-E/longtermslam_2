"""Exact feasible gauge in the WORKING plane model, not physical impossibility."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.material_supported_continuous_alignment_v1 import historical_offsets,pose_vector,material_residual


def main():
    root=Path('results/material_supported_continuous_alignment_v2');cfg=read(root/'config.json');manifest=read(root/'manifest.json');radius=manifest['motion_limits'][1];rows=[]
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (dest/'features.pkl').open('rb') as f:features=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        pred=read(dest/'predictions.json');selection=pred['material_shared']['selection']
        if not selection:continue
        for i,k,l in selection:
            chosen=[candidates[i][0][k],candidates[i][1][l]];slots=inp['T0'][:,:3,3];ff=[features[t+1][b['slot']] for t,b in enumerate(chosen)];reference=historical_offsets(features[0][i])
            sets=[set(f['axis'] for f in fff) for fff in ff];zkeys=sets[1]&{2,5}
            free=not bool(zkeys&set(reference)) and not bool(zkeys&sets[0])
            x=np.concatenate([pose_vector(b['T'],slots[b['slot']]) for b in chosen]);R=Rotation.from_rotvec(x[6:9]).as_matrix();v=R[:,2];u=x[9:12]
            dot=float(u@v);disc=max(0.,dot*dot+radius*radius-float(u@u));interval=np.array([-dot-np.sqrt(disc),-dot+np.sqrt(disc)]);width=float(interval[1]-interval[0]);r,J=material_residual(x,ff,reference,cfg,True)
            max_error=None
            if free:
                delta=np.zeros(12);delta[9:]=v;assert np.linalg.norm(J@delta)<1e-7
                errors=[]
                for eps in interval*.999999:
                    trial=x+eps*delta;rr,_=material_residual(trial,ff,reference,cfg,True)
                    assert np.linalg.norm(trial[9:])<=radius+1e-10
                    errors.append(float(np.max(np.abs(rr-r))) if len(r) else 0.)
                max_error=max(errors);assert max_error<1e-7
            rows.append({'scene':tag,'asset':i,'axis_z_model_gauge':free,'feasible_epsilon_interval_m':interval.tolist(),'anchor_displacement_diameter_in_working_model_m':width if free else None,
                'residual_invariance_max_error':max_error,'born_z_axes_in_epoch2':sorted(zkeys),
                'interpretation':'Fixed R2, change u2 by epsilon*R2*ez, absorb new z-plane offset. Same profile residual and metadata ball feasibility. This proves an ambiguity of the compressed plane model, not of full RGB-D or actual physical material.'})
    widths=[r['anchor_displacement_diameter_in_working_model_m'] for r in rows if r['axis_z_model_gauge']]
    summary={'selected_assets':len(rows),'exact_z_gauge_assets':len(widths),'gauge_diameter_gt_5cm':sum(w>.05 for w in widths),'median_diameter_m':float(np.median(widths)) if widths else None,
        'max_residual_invariance_error':max(r['residual_invariance_max_error'] for r in rows if r['axis_z_model_gauge']) if widths else None}
    save(root/'depth_gauge_check.json',{'summary':summary,'GT_used':False,'predictions_changed':False,'rows':rows,'source_sha256':sha(Path(__file__))});print(summary,flush=True)


if __name__=='__main__':main()
