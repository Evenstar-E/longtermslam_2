"""Evaluate against independent annotations only AFTER pose predictions are sealed."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.run_rigid_identity_cut_ablation_v1 import load_problem
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem

def main():
    out=Path('results/ycbineoat_motion_v1');root=Path('data/ycbineoat_motion_prepared_v2')
    for name,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/name)==h
    public=read(root/'public.json');clouds={}
    for row in public['clouds']:
        with np.load(root/row['path']) as z:clouds[row['stem']]={'xyz':z['xyz'],'rgb':z['rgb']}
    rows=[];energy_rows=[]
    for folder in sorted(out.glob('pair_*')):
        pred=read(folder/'predictions.json');pair=pred['pair'];a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4)
        for T in [a,b]:assert np.isfinite(T).all() and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) and abs(np.linalg.det(T[:3,:3])-1)<1e-4
        gt=b@np.linalg.inv(a);c=np.array(pred['historical_observed_anchor_m']);target=gt[:3,:3]@c+gt[:3,3]
        for method in pred['methods']+[{'method':'bounded_candidate','T_current_from_history':pred['candidate']['poses'][0]}]:
            T=np.array(method['T_current_from_history']);error=float(np.linalg.norm(T[:3,:3]@c+T[:3,3]-target));angle=float(np.rad2deg(Rotation.from_matrix(T[:3,:3]@gt[:3,:3].T).magnitude()))
            rows.append({'pair':folder.name,'method':method['method'],'anchor_error_m':error,'rotation_error_deg':angle,'anchor_error_at_most_1cm':error<=.01,'GT_anchor_camera_displacement_norm_m':float(np.linalg.norm(target-c)),'annotation_frame':'Relative camera-coordinate object transform; no independent world-motion G assumed'})
        H=clouds[pair['history_stem']];C=clouds[pair['current_stem']];problem=CachedJointProblem([H],[C],np.zeros((1,1)),read('configs/rigid_identity_bounds_v1.json'))
        Egt=problem.energy(np.array([0]),[gt]);UB=pred['candidate']['energy'];slack=problem.cfg['near_optimal_energy_slack'];energy_rows.append({'pair':folder.name,'GT_pose_energy':Egt,'candidate_UB':UB,'slack':slack,'reference_pose_outside_candidate_near_optimal_energy_set':Egt>UB+slack+1e-7,'GT_pose_in_declared_domain':problem.in_domain(gt,0),'hist_GT_origin_z_m':float(a[2,3]),'current_GT_origin_z_m':float(b[2,3])})
    methods=sorted({r['method'] for r in rows});summary=[]
    for method in methods:
        rr=[r for r in rows if r['method']==method];error=np.array([r['anchor_error_m'] for r in rr]);summary.append({'method':method,'pairs':len(rr),'median_anchor_error_m':float(np.median(error)),'maximum_anchor_error_m':float(error.max()),'within_1cm':int((error<=.01).sum()),'median_rotation_error_deg':float(np.median([r['rotation_error_deg'] for r in rr]))})
    save(out/'scores.json',{'summary':summary,'rows':rows,'surrogate_energy_falsification':energy_rows,'scope':'12 deterministic correlated pairs of ONE real manipulation sequence, privileged masks and known identity. Independent annotation reference, not metrology-perfect truth, not cross-day or multi-identity novelty confirmation.'})
    print(summary);print('GT_OUTSIDE_ENERGY_SET',sum(r['reference_pose_outside_candidate_near_optimal_energy_set'] for r in energy_rows),'/',len(energy_rows))

if __name__=='__main__':main()
