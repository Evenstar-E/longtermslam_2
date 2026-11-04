"""Visible sampling means vs full-cell means; target-excluded point surfaces.

The second-moment identity explicitly checks equivalence to ordinary rendering.
"""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.ray_material_evidence_v1 import ray_states


def grouped_mean(x, labels, n):
    count=np.bincount(labels,minlength=n)
    total=np.column_stack([np.bincount(labels,weights=x[:,k],minlength=n) for k in range(x.shape[1])])
    return np.divide(total,count[:,None],out=np.zeros(total.shape,dtype=float),where=count[:,None]>0),count


def fuse_surface(frames_and_poses,origin,cfg):
    # All input poses map the history frame into the input frame.
    xyz=[];rgb=[]
    for frame,T in frames_and_poses:
        xyz.append(moved(frame.xyz,np.linalg.inv(T)));rgb.append(frame.rgb)
    x=np.concatenate(xyz);c=np.concatenate(rgb)
    _,inv=np.unique(np.floor((x-origin)/cfg['fine_voxel_m']).astype(int),axis=0,return_inverse=True)
    n=int(inv.max())+1;x,_=grouped_mean(x,inv,n);c,_=grouped_mean(c,inv,n)
    keys,cell=np.unique(np.floor((x-origin)/cfg['material_cell_m']).astype(int),axis=0,return_inverse=True)
    center,count=grouped_mean(x,cell,len(keys));v=np.bincount(cell,weights=np.sum((x-center[cell])**2,axis=1),minlength=len(keys))/count/3
    return dict(x=x,rgb=c,cell=cell,centers=center,variance=v,origin=np.asarray(origin),cell_keys=keys)


def front_pixels(x,K,shape):
    """One frontmost projected sample per pixel, no target depth in rendering."""
    uv=x@K.T;uv=np.divide(uv[:,:2],uv[:,2,None],out=np.full((len(x),2),-1e8),where=np.abs(uv[:,2,None])>1e-12)
    u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
    valid=(x[:,2]>0)&(u>=0)&(u<shape[1])&(v>=0)&(v<shape[0]);ids=np.flatnonzero(valid)
    pixel=v[ids]*shape[1]+u[ids]
    order=np.lexsort((ids,x[ids,2],pixel));idx=ids[order];pid=pixel[order]
    keep=np.r_[True,pid[1:]!=pid[:-1]] if len(pid) else np.zeros(0,bool)
    return idx[keep],u[idx[keep]],v[idx[keep]]


def residual_moments(observed,predicted,full_centers,cell,ncells,sigma2,variance):
    obsmean,count=grouped_mean(observed,cell,ncells);predmean,_=grouped_mean(predicted,cell,ncells)
    r=observed-predicted;mean_r,_=grouped_mean(r,cell,ncells)
    pixel=float(np.sum(r*r)/sigma2)
    mean=float(np.sum(count*np.sum(mean_r*mean_r,axis=1))/sigma2)
    scatter=float(np.sum((r-mean_r[cell])**2)/sigma2)
    fixed2=np.sum((obsmean-full_centers)**2,axis=1)
    fixed=float(np.sum(count*fixed2)/sigma2)
    inflated=float(np.sum(count*fixed2/(sigma2+variance)))
    shift=np.linalg.norm(predmean-full_centers,axis=1)
    return dict(pixel=pixel,visible_mean=mean,scatter=scatter,fixed_mean=fixed,noise_inflated=inflated,
        equivalence_error=abs(pixel-mean-scatter),mean_shift_mm=float(np.average(shift[count>0],weights=count[count>0])*1000) if np.any(count) else 0.)


class Model:
    def __init__(self,surface,target,cfg):
        self.s=surface;self.c=target;self.cfg=cfg;self.tree=cKDTree(surface['x']);self.current_tree=cKDTree(target.xyz)
        self.invK=np.linalg.inv(target.K)

    def score(self,T,base):
        s=self.s;c=self.c;cfg=self.cfg;sigma2=cfg['sensor_sigma_m']**2
        x=moved(s['x'],T);ix,u,v=front_pixels(x,c.K,c.depth.shape);pred=x[ix]
        states=ray_states(pred,c.depth,c.mask,c.K,cfg['sensor_sigma_m'],[.001,5])[0]
        surf=states==1;free=states==2;n=max(len(ix),1)
        # Same observed/censored rays and colors for every centroid/pixel method.
        pix=np.column_stack([u[surf],v[surf],np.ones(sum(surf))]);obs=(pix@self.invK.T)*c.depth[v[surf],u[surf],None]
        observed_color=c.image[v[surf],u[surf]]
        col=float(np.sum(np.minimum(np.mean(np.abs(observed_color-s['rgb'][ix[surf]]),axis=1)/cfg['color_scale'],1)))
        common=float((sum(free)-sum(surf)+.5*col)/n)
        m=residual_moments(obs,pred[surf],moved(s['centers'],T),s['cell'][ix[surf]],len(s['centers']),sigma2,s['variance'])
        geometry_keys={'ordinary_rendered_pixels':'pixel','fixed_centroid':'fixed_mean','noise_inflated_centroid':'noise_inflated','visible_centroid':'visible_mean'}
        scores={'pair_only':float(base)}
        for name,key in geometry_keys.items():scores[name]=float(base+cfg['surface_weight']*(common+.5*m[key]/n))
        scores['visible_centroid_plus_scatter']=float(base+cfg['surface_weight']*(common+.5*(m['visible_mean']+m['scatter'])/n))
        # Ordinary fused-surface geometry, no ray censoring; same observed surface.
        fwd=self.current_tree.query(x)[0];reverse=self.tree.query(moved(c.xyz,np.linalg.inv(T)))[0]
        nn=float((np.mean(np.minimum(fwd/cfg['sensor_sigma_m'],1)**2)+np.mean(np.minimum(reverse/cfg['sensor_sigma_m'],1)**2))/2)
        scores['ordinary_shared_nn']=float(base+cfg['surface_weight']*nn)
        return scores,dict(rendered_pixels=len(ix),surface_pixels=int(sum(surf)),free_pixels=int(sum(free)),unknown_pixels=int(sum(states==0)),shared_nn=nn,moments=m)
