"""Vectorized exact same triangle projection, with independent scalar parity."""
import numpy as np


def closest_batch(points,triangles,valid,fallback):
    a,b,c=triangles[:,:,0],triangles[:,:,1],triangles[:,:,2];u=b-a;v=c-a;w=points[:,None]-a
    uu=(u*u).sum(2);vv=(v*v).sum(2);uv=(u*v).sum(2);uw=(u*w).sum(2);vw=(v*w).sum(2)
    den=uu*vv-uv*uv;s=(vv*uw-uv*vw)/np.maximum(den,1e-30);t=(uu*vw-uv*uw)/np.maximum(den,1e-30)
    projected=a+s[:,:,None]*u+t[:,:,None]*v;inside=(s>=0)&(t>=0)&(s+t<=1)&(den>1e-20)&valid
    candidates=[projected];distances=[np.where(inside,((projected-points[:,None])**2).sum(2),np.inf)]
    for x,y in [(a,b),(b,c),(c,a)]:
        d=y-x;lam=np.clip(((points[:,None]-x)*d).sum(2)/np.maximum((d*d).sum(2),1e-30),0,1)
        p=x+lam[:,:,None]*d;candidates.append(p);distances.append(np.where(valid,((p-points[:,None])**2).sum(2),np.inf))
    dist=np.stack(distances,axis=1);ix=dist.reshape(len(points),-1).argmin(1);kind=ix//triangles.shape[1];triangle=ix%triangles.shape[1]
    out=np.stack(candidates,axis=1)[np.arange(len(points)),kind,triangle]
    out[~valid.any(1)]=fallback[~valid.any(1)]
    return out
