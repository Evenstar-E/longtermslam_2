"""Offline, privileged-background diagnostic; not a measured camera-pose truth."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.real_temporal_v1 import cloud,fit_stats

def main():
    rows=[];reg=o3d.pipelines.registration
    for scene in ['scene_a','scene_b','scene_c']:
        clouds=[]
        for k in [0,1]:
            data=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_{k}.ply')
            mask=np.isin(data['class_idx'],[1,2])
            p=cloud(np.column_stack([data[n][mask] for n in ['x','y','z']])).voxel_down_sample(.08)
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.24,max_nn=40));clouds.append(p)
        a,b=clouds
        fit=reg.registration_icp(a,b,.3,np.eye(4),reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=100))
        T=np.asarray(fit.transformation);rot=float(np.degrees(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1))))
        rows.append(dict(scene=scene,T_current_from_history=T.tolist(),translation_m=float(np.linalg.norm(T[:3,3])),rotation_deg=rot,
            identity_fit=fit_stats(np.asarray(a.points),np.asarray(b.points),np.eye(4),.075),refined_fit=fit_stats(np.asarray(a.points),np.asarray(b.points),T,.075),
            caution='Oracle wall/floor classes; single local ICP, not an independently calibrated global frame or a certified pose.'))
    Path('results/real_temporal_v1/global_frame_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
