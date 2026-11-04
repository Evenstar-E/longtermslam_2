"""Auditable GT projection diagnostics, not a learned detector or visibility ground truth."""
import numpy as np

def quaternion_rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:raise ValueError('Invalid quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def rigid_matrix(t,q):
    T=np.eye(4);T[:3,:3]=quaternion_rotation(q);T[:3,3]=t
    return T

def world_mesh(points,bbox):
    q=bbox['world_R_center'];R=quaternion_rotation([q['x'],q['y'],q['z'],q['w']])
    return np.asarray(points)@R.T+np.array(bbox['world_P_center'])

def projected_surface(points,T_world_camera,K,width,height,min_depth=.1):
    camera=(points-T_world_camera[:3,3])@T_world_camera[:3,:3]
    valid=np.isfinite(camera).all(axis=1)&(camera[:,2]>min_depth)
    camera=camera[valid]
    if not len(camera):return np.empty((0,2),int),np.empty(0)
    uv=(camera@K.T);uv=uv[:,:2]/uv[:,2,None]
    inside=(uv[:,0]>=-.5)&(uv[:,0]<width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<height-.5)
    xy=np.floor(uv[inside]+.5).astype(int);z=camera[inside,2]
    if not len(xy):return xy,np.empty(0)
    # One nearest expected vertex per raster pixel: back surfaces cannot multiply votes.
    pixel=xy[:,1]*width+xy[:,0];order=np.lexsort((z,pixel));pixel=pixel[order]
    keep=np.r_[True,pixel[1:]!=pixel[:-1]];chosen=order[keep]
    return xy[chosen],z[chosen]

def depth_evidence(xy,z,depth,semantics,label,tolerance,cfg):
    n=len(z)
    if not n:
        return {'projected_pixels':0,'geometry_support':0,'class_support':0,'free':0,'occluded':0,'invalid':0,'class_mismatch':0,'state':'insufficient_projection'}
    d=depth[xy[:,1],xy[:,0]];classes=semantics[xy[:,1],xy[:,0]]
    valid=np.isfinite(d)&(d>0)
    support=valid&(np.abs(d-z)<=tolerance)
    same=support&(classes==label);free=valid&(d>z+tolerance);blocked=valid&(d<z-tolerance)
    counts={'projected_pixels':n,'geometry_support':int(support.sum()),'class_support':int(same.sum()),
       'free':int(free.sum()),'occluded':int(blocked.sum()),'invalid':int((~valid).sum()),'class_mismatch':int((support&~same).sum())}
    if n<cfg['minimum_projected_pixels']:state='insufficient_projection'
    elif counts['invalid']/n>=cfg['invalid_fraction']:state='invalid_depth'
    elif counts['class_support']>=cfg['minimum_support_pixels'] and counts['class_support']/n>=cfg['support_fraction']:state='supported_surface'
    elif counts['free']>=cfg['minimum_support_pixels'] and counts['free']/n>=cfg['free_fraction'] and counts['class_support']/n<cfg['free_max_support_fraction']:state='free_space_evidence'
    elif counts['occluded']/n>=cfg['occluded_fraction']:state='occluded'
    else:state='mixed_or_unresolved'
    return {**counts,'state':state}
