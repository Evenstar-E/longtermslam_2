"""Posthoc ALL-pair oracle diagnosis; GT is diagnostic only, not proposal selection."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem

root=Path('data/ycbineoat_motion_prepared_v2');out=Path('results/ycbineoat_motion_v1');cfg=read('configs/rigid_identity_bounds_v1.json');K=np.loadtxt(root/'public/cam_K.txt').reshape(3,3);rows=[];controls=[]
for folder in sorted(out.glob('pair_*')):
    pred=read(folder/'predictions.json');pair=pred['pair'];h=pair['history_stem'];c=pair['current_stem']
    with np.load(root/'public/clouds'/(h+'.npz')) as z:H={'xyz':z['xyz'],'rgb':z['rgb']}
    with np.load(root/'public/clouds'/(c+'.npz')) as z:C={'xyz':z['xyz'],'rgb':z['rgb']}
    p=CachedJointProblem([H],[C],np.zeros((1,1)),cfg);ha=np.loadtxt(root/'evaluation/poses'/(h+'.txt')).reshape(4,4);ca=np.loadtxt(root/'evaluation/poses'/(c+'.txt')).reshape(4,4);GT=ca@np.linalg.inv(ha)
    depth=np.array(Image.open(root/'public/depth'/(h+'.png'))).astype(float)/1000;mask=np.array(Image.open(root/'public/masks'/(h+'.png')))>0
    if mask.ndim==3:mask=mask.any(2)
    poses={'GT_reference':GT,'candidate':np.array(pred['candidate']['poses'][0])}
    supplied=min((r for r in pred['methods'] if r['inside_declared_domain']),key=lambda r:r['surrogate_energy']);poses['ordinary_best_supplied_by_same_energy']=np.array(supplied['T_current_from_history'])
    for method,T in poses.items():
        q=(C['xyz']-T[:3,3])@T[:3,:3];d=cKDTree(H['xyz']).query(q)[0];z=q[:,2];uv=(q@K.T);uv=uv[:,:2]/uv[:,2:];u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
        inside=(z>0)&(u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0]);depth_at=np.zeros(len(q));object_at=np.zeros(len(q),bool)
        depth_at[inside]=depth[v[inside],u[inside]];object_at[inside]=mask[v[inside],u[inside]]
        known=inside&(depth_at>.001);behind=known&(z>depth_at+.015);near=known&object_at&(np.abs(z-depth_at)<=.015);front=known&(z<depth_at-.015)
        geometry=cfg['geometry_weight']*float(np.mean(np.minimum(d/cfg['distance_scale_m'],1)**2));pcur=T[:3,:3]@p.anchor[0]+T[:3,3];motion=cfg['motion_weight']*min(float(np.linalg.norm(pcur-p.anchor[0])),1.)
        error=float(np.linalg.norm(pcur-(GT[:3,:3]@p.anchor[0]+GT[:3,3])))
        rows.append({'pair':folder.name,'method':method,'anchor_error_m':error,'geometry_term':geometry,'appearance_term':float(p.appearance[0,0]),'motion_regularizer':motion,'energy':p.energy(np.array([0]),[T]),'current_points':len(q),
            'historical_ray_behind_measured_depth_fraction':float(behind.mean()),'historical_ray_near_object_depth_fraction':float(near.mean()),'historical_ray_in_front_of_depth_fraction':float(front.mean()),'outside_or_unknown_depth_fraction':float((~known).mean()),
            'mean_clipped_geometric_loss_of_behind_points':None if not behind.any() else float(np.mean(np.minimum(d[behind]/cfg['distance_scale_m'],1)**2)),
            'scope':'GT-conditioned diagnostic, not an implemented new visibility likelihood or autonomous observation selector'})
        if method=='ordinary_best_supplied_by_same_energy':controls.append({'pair':folder.name,'method_selected':supplied['method'],'anchor_error_m':error,'energy':supplied['surrogate_energy'],'candidate_energy':pred['candidate']['energy']})
save(out/'energy_diagnosis.json',{'rows':rows,'ordinary_same_supplied_proposal_control':controls,'control_scope':'POSTHOC control after aggregate scores, no additional inference calls; selection by stored surrogate energy, not GT. Not independent confirmation.',
    'limitation':'Ray categories are a diagnostic from actual historical depth and supplied mask. Behind means not seen in that historical ray, not proof of complete object invisibility or exact surface correspondence.'})
print('ORDINARY_SAME_PROPOSALS',sum(r['anchor_error_m']<=.01 for r in controls),'/',len(controls),'median',float(np.median([r['anchor_error_m'] for r in controls])))
print('PAIR08',[r for r in rows if r['pair']=='pair_08'])
