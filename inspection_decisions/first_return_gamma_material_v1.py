"""Piecewise optical hazard, exact-distance approximation. No truth inputs.

The Gamma integral is exact for declared sufficient statistics, not for noisy
depth or opaque surface physics. Grid boundary events are not differentiated.
"""
import numpy as np
from scipy.special import gammaln, betaln


def log_gamma(a,b,n,L):
    return gammaln(a+n)-gammaln(a)+a*np.log(b)-(a+n)*np.log(b+L)


def log_beta(a,b,n,m):
    return betaln(a+n,b+m)-betaln(a,b)


def frame_rays(frame,cfg,owner):
    z=frame['depth'];h,w=z.shape;y,x=np.mgrid[:h,:w]
    r=np.c_[((x.ravel()-(w-1)/2)/cfg['focal_px']),((y.ravel()-(h-1)/2)/cfg['focal_px']),np.ones(h*w)]@frame['C'][:3,:3].T
    scale=np.linalg.norm(r,axis=1)
    return {'origin':frame['C'][:3,3], 'direction':r/scale[:,None],
            'distance':z.ravel()*scale, 'owner':owner.ravel(),
            'depth_jacobian_log':np.log(scale), 'pixel_count':h*w}


def ray_stats(O,D,s,hit,extent,h):
    """Exact slab/grid path lengths to observed first return (never beyond).

    Input rays must be unique; a ray crossing another object's ROI contributes
    survival exposure but no hit. Miss count is ordinary discrete occupancy's
    per-cell no-return count; partial censored cells count once in that control.
    """
    D=np.asarray(D,float).reshape(-1,3);O=np.broadcast_to(O,D.shape);s=np.asarray(s,float);hit=np.asarray(hit,bool)
    q=int(round(2*extent/h));V=q**3
    with np.errstate(divide='ignore',invalid='ignore'):
        inv=np.where(abs(D)>1e-14,1/D,np.inf)
        aa=(-extent-O)*inv;bb=(extent-O)*inv
    parallel=abs(D)<=1e-14
    aa=np.where(parallel,-np.inf,aa);bb=np.where(parallel,np.inf,bb)
    outside=(parallel&((O < -extent)|(O > extent))).any(1)
    lo=np.maximum(np.minimum(aa,bb).max(1),0);hi=np.minimum(np.maximum(aa,bb).min(1),s)
    valid=(hi>lo)&~outside&np.isfinite(s)&(s>0)
    endpoint=O+D*s[:,None];own=hit&np.isfinite(s)&(s>0)
    inside=((endpoint>=-extent)&(endpoint<extent)).all(1)
    if np.any(own&~inside):raise ValueError('Target hit outside declared material workspace')
    cell=np.clip(np.floor((endpoint[own]+extent)/h).astype(int),0,q-1)
    hitid=np.ravel_multi_index(cell.T,(q,q,q)) if len(cell) else np.array([],int)
    n=np.bincount(hitid,minlength=V).astype(float);L=np.zeros(V);m=np.zeros(V)
    if valid.any():
        o=O[valid];d=D[valid];lower=lo[valid];upper=hi[valid]
        planes=-extent+np.arange(1,q)*h
        with np.errstate(divide='ignore',invalid='ignore'):
            cross=((planes[None,None,:]-o[:,:,None])/d[:,:,None]).reshape(len(o),-1)
        cross=np.nan_to_num(cross,nan=np.inf)
        cross=np.clip(cross,lower[:,None],upper[:,None]);times=np.sort(np.c_[lower,cross,upper],axis=1)
        lengths=np.diff(times,axis=1);rr,cc=np.where(lengths>1e-13)
        mid=(times[rr,cc]+times[rr,cc+1])/2
        xyz=o[rr]+d[rr]*mid[:,None];ind=np.clip(np.floor((xyz+extent)/h).astype(int),0,q-1);ids=np.ravel_multi_index(ind.T,(q,q,q))
        L=np.bincount(ids,weights=lengths[rr,cc],minlength=V)
        # One cell visit per straight ray. The endpoint cell with own hit is
        # occupied, not also an empty Bernoulli trial.
        vh=hit[valid];ep=endpoint[valid];hc=np.clip(np.floor((ep+extent)/h).astype(int),0,q-1);hid=np.ravel_multi_index(hc.T,(q,q,q))
        empty=~(vh[rr]&(ids==hid[rr]));m=np.bincount(ids[empty],minlength=V).astype(float)
    return np.stack([n,L,m]),{'rays':len(s),'intersecting':int(valid.sum()),'hits':int(own.sum()),'exposure_m':float(L.sum()),'outside_hits':0}


def stats(frames,T,slot,cfg):
    total=np.zeros((3,int(round(2*cfg['hazard_extent_m']/cfg['hazard_voxel_m']))**3));log=[]
    for f in frames:
        O=(f['origin']-T[:3,3])@T[:3,:3];D=f['direction']@T[:3,:3]
        value,meta=ray_stats(O,D,f['distance'],f['owner']==slot,cfg['hazard_extent_m'],cfg['hazard_voxel_m']);total+=value;log.append(meta)
    return total,log


def posterior(H,cfg):
    return (cfg['gamma_shape']+H[0],cfg['gamma_rate_m']+H[1],cfg['beta_alpha']+H[0],cfg['beta_beta']+H[2])


def unary(H,Q,cfg,method):
    a,b,alpha,beta=posterior(H,cfg);n,L,m=np.moveaxis(Q,-2,0)
    if method in ['gamma_joint','gamma_independent']:return -log_gamma(a,b,n,L).sum(-1)
    if method=='gamma_mean':return (a/b*L-n*np.log(a/b)).sum(-1)
    if method=='beta_joint':return -log_beta(alpha,beta,n,m).sum(-1)
    raise ValueError(method)


def pair_costs(H,A,B,cfg,method):
    """Exact sparse overlap correction; disjoint supports factorize exactly."""
    u=unary(H,A,cfg,method);v=unary(H,B,cfg,method);cost=u[:,None]+v[None,:]
    if method in ['gamma_independent','gamma_mean']:return cost
    a,b,alpha,beta=posterior(H,cfg)
    for k in range(A.shape[-1]):
        ia=np.flatnonzero((A[:,0,k]+A[:,1,k])>0);ib=np.flatnonzero((B[:,0,k]+B[:,1,k])>0)
        if not len(ia) or not len(ib):continue
        n=A[ia,0,k,None];p=B[None,ib,0,k]
        if method=='gamma_joint':
            L=A[ia,1,k,None];M=B[None,ib,1,k];f=lambda x,y:log_gamma(a[k],b[k],x,y)
        else:
            L=A[ia,2,k,None];M=B[None,ib,2,k];f=lambda x,y:log_beta(alpha[k],beta[k],x,y)
        cost[np.ix_(ia,ib)]-=f(n+p,L+M)-f(n,L)-f(p,M)
    return cost


def scalar_cost(H,A,B,cfg,method):
    if method=='gamma_joint':return float(unary(H,A+B,cfg,method))
    if method=='beta_joint':return float(unary(H,A+B,cfg,method))
    return float(unary(H,A,cfg,method)+unary(H,B,cfg,method))
