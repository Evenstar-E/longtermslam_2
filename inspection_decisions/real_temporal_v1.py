"""ID-blind point-cloud registration modules. No evaluation IDs or GT file access."""
import subprocess, time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def cloud(x,colors=None):
    p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(np.asarray(x,float))
    if colors is not None:p.colors=o3d.utility.Vector3dVector(colors)
    return p

def prepare(x,cfg):
    p=cloud(x).voxel_down_sample(cfg['voxel_m'])
    if len(p.points)>cfg['maximum_points']:
        p=p.select_by_index(np.linspace(0,len(p.points)-1,cfg['maximum_points']).astype(int))
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['normal_radius_m'],max_nn=30))
    f=o3d.pipelines.registration.compute_fpfh_feature(p,o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['feature_radius_m'],max_nn=100))
    return p,f

def moved(x,T):return np.asarray(x)@T[:3,:3].T+T[:3,3]

def fit_stats(a,b,T,threshold=.075):
    x=moved(a,T);da=cKDTree(b).query(x)[0];db=cKDTree(x).query(b)[0]
    return {'cost':float((np.minimum(da/threshold,1)**2).mean()/2+(np.minimum(db/threshold,1)**2).mean()/2),
            'support_a':float((da<threshold).mean()),'support_b':float((db<threshold).mean()),
            'rmse_clipped_m':float(np.sqrt((np.minimum(da,threshold)**2).mean()/2+(np.minimum(db,threshold)**2).mean()/2)),
            'median_forward_m':float(np.median(da)),'median_reverse_m':float(np.median(db))}

def fpfh_correspondences(a,b,fa,fb,cfg):
    aa=np.asarray(fa.data).T;bb=np.asarray(fb.data).T
    ab=cKDTree(bb).query(aa)[1];ba=cKDTree(aa).query(bb)[1]
    ix=np.flatnonzero(ba[ab]==np.arange(len(aa)))
    if len(ix)>cfg['teaser_max_correspondences']:
        ix=ix[np.linspace(0,len(ix)-1,cfg['teaser_max_correspondences']).astype(int)]
    return np.asarray(a.points)[ix],np.asarray(b.points)[ab[ix]]

def teaser(a,b,cfg,work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    source=work/'matches.txt';target=work/'transform.txt'
    if len(a)<3: return None,{'reason':'fewer_than_three_mutual_features'}
    np.savetxt(source,np.column_stack([a,b]),header=str(len(a)),comments='')
    # Windows -> WSL mapped paths; none of these files contain evaluator truth.
    def linux(path):
        s=str(Path(path).resolve());return '/mnt/'+s[0].lower()+s[2:].replace('\\','/')
    command=['wsl','-d','Ubuntu-20.04','--exec','env','OMP_NUM_THREADS=1',
             linux('tools/real_registration_v1/build/teaser_cli'),linux(source),linux(target),str(cfg['teaser_noise_bound_m'])]
    try:
        r=subprocess.run(command,capture_output=True,timeout=35)
        (work/'stdout.bin').write_bytes(r.stdout);(work/'stderr.bin').write_bytes(r.stderr)
        if r.returncode!=0:return None,{'reason':'teaser_exit','returncode':r.returncode}
        T=np.loadtxt(target)
        if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid transform')
        return T,{'mutual_features':len(a),'implementation':'unchanged TEASER++ linked library f91cfdb7'}
    except subprocess.TimeoutExpired:
        return None,{'reason':'timeout_35s'}

def register(a,b,fa,fb,method,cfg,work,initial=None):
    o3d.utility.random.seed(cfg['seed']);start=time.perf_counter();reg=o3d.pipelines.registration
    if method=='centroid_hungarian':
        T=np.eye(4);T[:3,3]=np.asarray(b.points).mean(0)-np.asarray(a.points).mean(0)
        extra={}
    elif method=='centroid_icp':
        T=np.eye(4);T[:3,3]=np.asarray(b.points).mean(0)-np.asarray(a.points).mean(0)
        result=reg.registration_icp(a,b,cfg['gicp_distance_m'],T,reg.TransformationEstimationPointToPoint(),reg.ICPConvergenceCriteria(max_iteration=cfg['gicp_iterations']))
        T=np.asarray(result.transformation);extra={}
    elif method in ('fgr','fgr_gicp'):
        result=reg.registration_fgr_based_on_feature_matching(a,b,fa,fb,reg.FastGlobalRegistrationOption(maximum_correspondence_distance=cfg['registration_distance_m'],iteration_number=64))
        T=np.asarray(result.transformation);extra={}
        if method=='fgr_gicp':
            result=reg.registration_generalized_icp(a,b,cfg['gicp_distance_m'],T,reg.TransformationEstimationForGeneralizedICP(),reg.ICPConvergenceCriteria(max_iteration=cfg['gicp_iterations']))
            T=np.asarray(result.transformation)
    elif method=='teaser':
        x,y=fpfh_correspondences(a,b,fa,fb,cfg);T,extra=teaser(x,y,cfg,work)
    elif method=='gicp_initial':
        result=reg.registration_generalized_icp(a,b,cfg['gicp_distance_m'],initial,reg.TransformationEstimationForGeneralizedICP(),reg.ICPConvergenceCriteria(max_iteration=cfg['gicp_iterations']))
        T=np.asarray(result.transformation);extra={}
    else:raise ValueError(method)
    if T is None:return None,dict(extra,elapsed_s=time.perf_counter()-start,cost=1.0)
    return T,dict(fit_stats(np.asarray(a.points),np.asarray(b.points),T,cfg['registration_distance_m']),**extra,elapsed_s=time.perf_counter()-start)

def assign(cost,cfg):
    # Each observation may be unmatched; no forced one-to-one match to a wrong asset.
    n,m=cost.shape;aug=np.concatenate([cost,np.full((n,n),cfg['null_assignment_cost'])],axis=1)
    ii,jj=linear_sum_assignment(aug)
    return {int(i):int(j) if j<m else None for i,j in zip(ii,jj)}
