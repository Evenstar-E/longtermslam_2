"""Post-seal loss decomposition and local-solver failure trace, NOT new inference."""
from pathlib import Path
import numpy as np
import open3d as o3d
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.run_real_continuous_proposals_v1 import cloud
from inspection_decisions.ray_material_evidence_v1 import Frame,ray_states


def main():
    root=Path('results/real_continuous_proposals_v1');cfg=read(root/'manifest.json')['config'];fc=read('results/ray_material_full_resolution_v1/effective_configuration.json')
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    decompositions=[];traces=[]
    for tag in ['pair_07','pair_08']:
        pred=read(root/tag/'predictions.json');pair=pred['pair'];h=Frame(cfg['public_root'],pair['history_stem'],fc);c=Frame(cfg['public_root'],pair['current_stem'],fc);anchor=np.array(pred['anchor'])
        if tag=='pair_07':
            i=pred['endpoints']['point_to_plane__old_nn'];correct=pred['pool'][i];T=np.array(correct['T_current_from_history']);query=(c.all_xyz-T[:3,3])@T[:3,:3]
            state,*_=ray_states(query,h.depth,h.mask,h.K,fc['depth_tolerance_m'],fc['depth_valid_interval_m'])
            choices={'accurate_point_to_plane':i,'selected_nn':pred['choices']['all__old_nn'],'selected_visibility':pred['choices']['all__ordinary_overlap_rgb']}
            for label,j in choices.items():
                e=pred['pool'][j];T=np.array(e['T_current_from_history']);q=(c.all_xyz-T[:3,3])@T[:3,:3];dist=h.tree.query(q,workers=1)[0];cost=.6*np.minimum(dist/.075,1)**2
                motion=.2*min(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-anchor),1.)
                row=dict(label=label,index=j,nn_geometry=float(cost.mean()),motion=float(motion),nn_total=float(cost.mean()+motion),
                         visibility=e['scores']['ordinary_overlap_rgb'],reference_groups={})
                assert abs(row['nn_total']-e['scores']['old_nn'])<1e-12
                for group,k in [('compatible_positive_under_refinement',1),('measured_absence_under_refinement',2),('unknown_under_refinement',0)]:
                    mask=state==k;row['reference_groups'][group]=dict(points=int(mask.sum()),fraction=float(mask.mean()),nn_contribution=float(cost[mask].sum()/len(cost)))
                decompositions.append(row)
        else:
            voxel=cfg['voxel_scales_m'][0];source=cloud(h,voxel,cfg);target=cloud(c,voxel,cfg)
            for seed in cfg['seeds']:
                initial=pred['choices']['old__'+seed];T=np.array(pred['pool'][initial]['T_current_from_history']);history=[]
                for step in range(8):
                    evaluation=o3d.pipelines.registration.evaluate_registration(source,target,voxel*cfg['correspondence_distance_voxel_multiplier'],T)
                    pairs=np.asarray(evaluation.correspondence_set);row=dict(iteration=step,correspondences=len(pairs),fitness=evaluation.fitness,T=T.tolist())
                    if len(pairs):
                        points=np.asarray(source.points)[pairs[:,0]]@T[:3,:3].T+T[:3,3];normals=np.asarray(target.normals)[pairs[:,1]]
                        center=points.mean(0);scale=max(float(np.linalg.norm(points-center,axis=1).mean()),1e-6)
                        J=np.c_[np.cross((points-center)/scale,normals),normals];eigen=np.linalg.eigvalsh(J.T@J/max(len(J),1));row['dimensionless_local_hessian_eigenvalues']=eigen.tolist();row['rotation_length_scale_m']=scale
                    history.append(row)
                    if not len(pairs):break
                    result=o3d.pipelines.registration.registration_icp(source,target,voxel*cfg['correspondence_distance_voxel_multiplier'],T,o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
                    next_T=np.asarray(result.transformation);row['one_step_translation_change_m']=float(np.linalg.norm(next_T[:3,3]-T[:3,3]))
                    row['one_step_anchor_change_m']=float(np.linalg.norm(next_T[:3,:3]@anchor+next_T[:3,3]-T[:3,:3]@anchor-T[:3,3]))
                    row['one_step_rotation_deg']=float(np.degrees(np.arccos(np.clip((np.trace(next_T[:3,:3]@T[:3,:3].T)-1)/2,-1,1))))
                    T=next_T.copy()
                    if not np.isfinite(T).all():row['nonfinite_next']=True;break
                traces.append(dict(pair=tag,seed=seed,history=history))
    save(root/'mechanism_diagnosis.json',dict(postseal=True,predictions_changed=False,GT_not_used_in_computations=True,
        failure_cases_identified_from_previous_GT_scores=True,pair07_loss_decomposition=decompositions,pair08_single_step_replay=traces,
        interpretation='Local Hessian conditioning is diagnostic, not a calibrated global pose/anchor certificate; ray states are operational raster compatibility, not verified same-material labels.',source_sha256=sha(Path(__file__))))
    print('LOSS',decompositions,flush=True)
    print('TRACE',[dict(seed=t['seed'],steps=[{k:v for k,v in r.items() if k!='T'} for r in t['history']]) for t in traces],flush=True)


if __name__=='__main__':main()
