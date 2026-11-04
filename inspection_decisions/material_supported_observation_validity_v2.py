"""Noise-resolved 2D support and camera-transversality check for measured planes.

An observation validity correction, not an algorithmic novelty or calibrated
normal-error certificate. Known SIM depth bound only; do not transfer to real
RGB-D without a sensor model. V1 sources/results remain untouched.
"""
import numpy as np
import open3d as o3d
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES


def validity(points,normal,centroid,camera,cfg):
    singular=np.linalg.svd(points-centroid,compute_uv=False)
    ray_max=np.sqrt(1+((cfg['width']-1)/(2*cfg['focal_px']))**2+((cfg['height']-1)/(2*cfg['focal_px']))**2)
    point_bound=cfg['depth_noise_bound_m']*ray_max
    # Centering has operator norm <=1. Weyl: a second singular value no larger
    # than this Frobenius perturbation bound does not resolve a 2D patch.
    frobenius_bound=point_bound*np.sqrt(len(points))
    rank_resolved=bool(len(singular)>1 and singular[1]>frobenius_bound)
    incidence=float(abs(normal@(camera-centroid)))
    transversal=incidence>point_bound
    return {'resolved':rank_resolved and transversal,'second_singular_value_m':float(singular[1]) if len(singular)>1 else 0.,
        'noise_frobenius_bound_m':float(frobenius_bound),'camera_plane_distance_m':incidence,'point_noise_bound_m':float(point_bound),
        'rank_resolved':rank_resolved,'transversal':transversal}


def extract_planes(points,camera,center,view,cfg,seed):
    p=np.asarray(points)-center;camera=np.asarray(camera)-center;remaining=np.arange(len(p));best={};records=[]
    o3d.utility.random.seed(int(seed))
    for attempt in range(cfg['planes_per_view_max']):
        if len(remaining)<cfg['plane_min_points']:break
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(p[remaining])
        _,inliers=cloud.segment_plane(cfg['plane_fit_threshold_m'],3,cfg['plane_ransac_iterations'])
        if len(inliers)<cfg['plane_min_points']:break
        selected=remaining[np.array(inliers)];xyz=p[selected];centroid=xyz.mean(0);_,_,V=np.linalg.svd(xyz-centroid,full_matrices=False);normal=V[-1]
        if normal@(camera-centroid)<0:normal=-normal
        k=int(np.argmax(AXES@normal));aligned=float(AXES[k]@normal);check=validity(xyz,normal,centroid,camera,cfg)
        record={'axis':k,'normal':normal,'centroid':centroid,'count':len(selected),'view':view,'axis_cosine':aligned,'rms_m':float(np.sqrt(np.mean(((xyz-centroid)@normal)**2))),'measured_indices':selected,'validity':check}
        records.append(record)
        if check['resolved'] and aligned>=cfg['plane_axis_cosine_min'] and (k not in best or len(selected)>best[k]['count']):best[k]=record
        mask=np.ones(len(remaining),bool);mask[np.array(inliers)]=False;remaining=remaining[mask]
    return list(best.values()),{'raw_points':len(p),'extracted':len(records),'retained':len(best),'remaining':len(remaining),
        'unresolved_plane_candidates':sum(not f['validity']['resolved'] for f in records),'rank_unresolved_candidates':sum(not f['validity']['rank_resolved'] for f in records),
        'camera_nontransversal_candidates':sum(not f['validity']['transversal'] for f in records)}
