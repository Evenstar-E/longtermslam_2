"""Bidirectional local motion fit and pose-marginalized information diagnostics.

All parts are candidate mechanisms built from established geometry/estimation.
The Schur diagnostic is NOT a calibrated posterior or a new invention by itself.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scripts.joint_estimation import transform
from scripts.partial_surface_estimation import signed_box_distance,perturb


def sample_faces(box,supported,n):
    p=[];normals=[]
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for side in (0,1):
            if not supported[2*axis+side]:continue
            a,b=np.meshgrid(np.linspace(box[0,other[0]],box[1,other[0]],n),np.linspace(box[0,other[1]],box[1,other[1]],n))
            q=np.empty((n*n,3));q[:,axis]=box[side,axis];q[:,other[0]]=a.ravel();q[:,other[1]]=b.ravel()
            normal=np.zeros(3);normal[axis]=2*side-1
            p.extend(q);normals.extend([normal]*len(q))
    return np.array(p),np.array(normals)


def project(p,T,obs):
    cp=transform(p,np.linalg.inv(T));h,w=obs['depth'].shape
    # Derive focal and centre from saved calibrated rays, without scene truth.
    focal=1/(obs['rays'][1,0]-obs['rays'][0,0])
    uv=cp[:,:2]/np.maximum(cp[:,2,None],1e-8)*focal+[(w-1)/2,(h-1)/2]
    return uv,cp[:,2],focal


def context(obs,i,j,cfg):
    q=obs['query'][i];pix=q['all_pixels'];camera_points=obs['rays'][pix]*obs['depth'].ravel()[pix,None]
    take=np.linspace(0,len(camera_points)-1,min(len(camera_points),cfg['current_max_points']),dtype=int)
    p,n=sample_faces(obs['bounds'][j],obs['face_support'][j],cfg['surface_samples_per_edge'])
    mask=np.zeros(obs['depth'].size,bool);mask[pix]=True;mask=mask.reshape(obs['depth'].shape)
    return dict(camera_points=camera_points,take=take,samples=p,normals=n,mask=mask,silhouette=distance_transform_edt(~mask))


def visibility(ctx,obs,T,delta,cfg,external):
    p=ctx['samples']+delta;front=np.sum((T[:3,3]-p)*ctx['normals'],axis=1)>0
    uv,z,_=project(p,T,obs);h,w=obs['depth'].shape
    inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    result=front&inside
    if external:
        x=np.clip(np.rint(uv[:,0]).astype(int),0,w-1);y=np.clip(np.rint(uv[:,1]).astype(int),0,h-1)
        measured=obs['depth'][y,x];own=ctx['mask'][y,x]
        result&=(measured>0)&(~((measured<z-cfg['occlusion_margin_m'])&~own))
    return result


def residual(ctx,obs,T,delta,j,selected,cfg,method):
    full=transform(ctx['camera_points'],T);p=full[ctx['take']]
    forward=signed_box_distance(p-delta,obs['bounds'][j])/np.sqrt(len(p))
    if method=='one_sided' or not selected.any():return forward
    model=ctx['samples'][selected]+delta;tree=cKDTree(full)
    reverse=tree.query(model)[0]/np.sqrt(len(model))
    uv,z,f=project(model,T,obs)
    sil=map_coordinates(ctx['silhouette'],[uv[:,1],uv[:,0]],order=1,mode='nearest')*z/f
    return np.r_[forward,reverse,cfg['silhouette_weight']*sil/np.sqrt(len(model))]


def stable_pose_information(obs,T,assignment,weights):
    """Information for the actual v1 surface-pose model, excluding moved groups."""
    selected=np.flatnonzero(weights>.5)
    if len(selected)==0:return np.zeros((6,6))
    def fun(v):
        trial=perturb(T,v)
        return np.concatenate([signed_box_distance(transform(obs['query'][i]['points'],trial),obs['bounds'][assignment[i]])/np.sqrt(len(obs['query'][i]['points'])) for i in selected])
    cols=[]
    for k in range(6):
        v=np.zeros(6);v[k]=.0001;cols.append((fun(v)-fun(-v))/.0002)
    J=np.array(cols).T
    return J.T@J


def diagnostic(ctx,obs,T,delta,j,selected,cfg,method,pose_information):
    cols=[]
    for k in range(9):
        step=cfg['jacobian_step_m'] if k>=6 else .0001;v=np.zeros(9);v[k]=step
        def value(v):return residual(ctx,obs,perturb(T,v[:6]),delta+v[6:],j,selected,cfg,method)
        cols.append((value(v)-value(-v))/(2*step))
    J=np.array(cols).T;H=J.T@J;conditional=H[6:,6:]
    marginal=conditional-H[6:,:6]@np.linalg.pinv(H[:6,:6]+pose_information,rcond=1e-8)@H[:6,6:]
    result={}
    for name,M in [('conditional',conditional),('marginal',marginal)]:
        eig,V=np.linalg.eigh((M+M.T)/2);eig=np.maximum(eig,0)
        threshold=max((cfg['information_noise_scale_m']/cfg['maximum_direction_scale_m'])**2,cfg['relative_eigen_threshold']*eig[-1])
        good=eig>threshold;P=V[:,good]@V[:,good].T
        result[name+'_eigenvalues']=eig.tolist();result[name+'_rank']=int(good.sum());result[name+'_projector']=P.tolist()
    return result


def fit_motion(obs,T,assignment,i,cfg,method,pose_information):
    j=assignment[i];ctx=context(obs,i,j,cfg);delta=np.zeros(3);bound=cfg['translation_bound_m']
    for _ in range(cfg['outer_visibility_iterations']):
        selected=visibility(ctx,obs,T,delta,cfg,method=='visible_bidirectional')
        # Gate frozen during inner optimization: moving behind an occluder cannot
        # instantly delete residuals during a finite-difference update.
        objective=lambda d:np.r_[residual(ctx,obs,T,d,j,selected,cfg,method),cfg['regularization']*d]
        fit=least_squares(objective,delta,bounds=(-bound,bound),loss='soft_l1',f_scale=.03/np.sqrt(len(ctx['take'])),max_nfev=cfg['max_nfev'])
        delta=fit.x
    selected=visibility(ctx,obs,T,delta,cfg,method=='visible_bidirectional')
    info=diagnostic(ctx,obs,T,delta,j,selected,cfg,method,pose_information)
    return dict(delta=delta.tolist(),near_bound=bool(np.max(np.abs(delta))>bound-.001),model_samples=int(selected.sum()),**info)
