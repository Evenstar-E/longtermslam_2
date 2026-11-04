"""Observed-surface continuous correspondence proposals; no truth/file access."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from inspection_decisions.textured_material_matching_v1 import world_points,rigid


class SurfaceMatcher(nn.Module):
    """Ordinary Siamese dense local correlation, deliberately a small prototype."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(10,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,48,3,padding=1),nn.GroupNorm(6,48),nn.SiLU(),nn.Conv2d(48,32,1))
        self.null=nn.Sequential(nn.Linear(66,32),nn.SiLU(),nn.Linear(32,1))

    def encode(self,patch):
        return F.normalize(self.encoder(patch),dim=1)

    def query(self,feature):
        # Center of an even 12x12 grid is between four samples.
        return F.normalize(feature[:,:,5:7,5:7].mean((2,3)),dim=1)

    def forward(self,source,target,mask):
        a=self.query(self.encode(source));b=self.encode(target).flatten(2).transpose(1,2)
        score=torch.einsum('bd,bvd->bv',a,b)/.12
        score=score.masked_fill(~mask,-1e4)
        den=mask.sum(1).clamp_min(1)
        mean=(b*mask[:,:,None]).sum(1)/den[:,None]
        context=torch.cat([a,mean,score.max(1).values[:,None],(mask.float().mean(1))[:,None]],1)
        return torch.cat([score,self.null(context)],1)


def surface_grids(frames,groups,cfg,v4):
    """12x12 measured depth grid at each metric patch, with witnessed triangles.

    Coordinates are actual nearest RGB-D samples (not the estimated tangent
    plane). No face is interpolated across a missing/other-slot/depth edge.
    """
    result=[];size=cfg['patch_size'];g=np.linspace(-1,1,size);gx,gy=np.meshgrid(g,g)
    for slot,a in enumerate(groups):
        n=len(a['xyz']);xyz=np.zeros((n,size*size,3));uv=np.zeros((n,size*size,2),int);valid=np.zeros((n,size*size),bool)
        for fi,f in enumerate(frames):
            ii=np.flatnonzero(a['frame']==fi)
            if not len(ii):continue
            p=a['uv'][ii];pix=np.rint(p).astype(int);h,w=f['depth'].shape;r=cfg['normal_window_px']
            dd=np.arange(-r,r+1);dx,dy=np.meshgrid(dd,dd)
            nx=np.clip(pix[:,0,None]+dx.ravel(),0,w-1);ny=np.clip(pix[:,1,None]+dy.ravel(),0,h-1)
            near=world_points(f,np.stack([nx,ny],-1).reshape(-1,2)).reshape(len(ii),-1,3)
            ok=(f['anonymous_slot'][ny,nx]==slot)&f['depth_valid'][ny,nx];wt=ok/np.maximum(ok.sum(1,keepdims=True),1)
            mean=(near*wt[:,:,None]).sum(1);delta=near-mean[:,None]
            cov=np.einsum('nki,nkj,nk->nij',delta,delta,wt);_,vv=np.linalg.eigh(cov);normal=vv[:,:,0]
            normal*=np.where(((f['C'][:3,3]-a['xyz'][ii])*normal).sum(1)>=0,1.,-1.)[:,None]
            axis=np.tile([1.,0,0],(len(ii),1));axis[abs(normal[:,0])>.9]=[0,1.,0]
            t1=axis-(axis*normal).sum(1)[:,None]*normal;t1/=np.maximum(np.linalg.norm(t1,axis=1,keepdims=True),1e-8);t2=np.cross(normal,t1)
            plane=a['xyz'][ii,None,None]+v4['surface_radius_m']*(gx[None,:,:,None]*t1[:,None,None]+gy[None,:,:,None]*t2[:,None,None])
            pc=(plane-f['C'][:3,3])@f['C'][:3,:3];z=pc[:,:,:,2]
            image=pc[:,:,:,:2]/np.maximum(z[:,:,:,None],1e-8)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2]
            image=np.rint(image).astype(int);inside=(z>0)&(image[:,:,:,0]>=0)&(image[:,:,:,0]<w)&(image[:,:,:,1]>=0)&(image[:,:,:,1]<h)
            image[:,:,:,0]=np.clip(image[:,:,:,0],0,w-1);image[:,:,:,1]=np.clip(image[:,:,:,1],0,h-1)
            x,y=image[:,:,:,0],image[:,:,:,1];depth=f['depth'][y,x]
            yes=inside&f['depth_valid'][y,x]&(f['anonymous_slot'][y,x]==slot)&(abs(depth-z)<.06)
            uv[ii]=image.reshape(len(ii),-1,2);xyz[ii]=world_points(f,image.reshape(-1,2)).reshape(len(ii),-1,3);valid[ii]=yes.reshape(len(ii),-1)
        triangles=[]
        for y in range(size-1):
            for x in range(size-1):
                k=y*size+x;triangles.extend([[k,k+1,k+size],[k+1,k+size+1,k+size]])
        triangles=np.array(triangles,int);tr=xyz[:,triangles]
        tri_valid=valid[:,triangles].all(2)
        tri_valid&=np.maximum.reduce([np.linalg.norm(tr[:,:,0]-tr[:,:,1],axis=2),np.linalg.norm(tr[:,:,1]-tr[:,:,2],axis=2),np.linalg.norm(tr[:,:,2]-tr[:,:,0],axis=2)])<=v4['surface_edge_limit_m']
        tri_valid&=np.linalg.norm(np.cross(tr[:,:,1]-tr[:,:,0],tr[:,:,2]-tr[:,:,0]),axis=2)>1e-9
        result.append(dict(xyz=xyz,uv=uv,valid=valid,triangles=triangles,tri_valid=tri_valid))
    return result


def closest_on_triangles(point,triangles):
    """Exact closest point using interior plane projection and all three edges."""
    if not len(triangles):return None
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];u=b-a;v=c-a;w=point-a
    uu=(u*u).sum(1);vv=(v*v).sum(1);uv=(u*v).sum(1);uw=(u*w).sum(1);vw=(v*w).sum(1)
    den=uu*vv-uv*uv;s=(vv*uw-uv*vw)/np.maximum(den,1e-30);t=(uu*vw-uv*uw)/np.maximum(den,1e-30)
    projected=a+s[:,None]*u+t[:,None]*v;inside=(s>=0)&(t>=0)&(s+t<=1)&(den>1e-20)
    candidates=[];distances=[]
    candidates.append(projected);distances.append(np.where(inside,((projected-point)**2).sum(1),np.inf))
    for x,y in [(a,b),(b,c),(c,a)]:
        d=y-x;lam=np.clip(((point-x)*d).sum(1)/np.maximum((d*d).sum(1),1e-30),0,1)
        p=x+lam[:,None]*d;candidates.append(p);distances.append(((p-point)**2).sum(1))
    dist=np.stack(distances);kind,index=np.unravel_index(dist.argmin(),dist.shape)
    return candidates[kind][index]


def decode_local(scores,grid,anchor,continuous,size=12):
    valid=grid['valid'][anchor]
    if not valid.any():return None,0.
    logits=np.where(valid,scores,-1e4);peak=int(logits.argmax());xyz=grid['xyz'][anchor]
    if not continuous:return xyz[peak],float(logits[peak])
    y,x=divmod(peak,size);yy,xx=np.mgrid[max(0,y-1):min(size,y+2),max(0,x-1):min(size,x+2)];ix=(yy*size+xx).ravel()
    ix=ix[valid[ix]];weight=np.exp(logits[ix]-logits[ix].max());mean=(xyz[ix]*weight[:,None]).sum(0)/weight.sum()
    tri=grid['triangles'];take=grid['tri_valid'][anchor]&np.isin(tri,ix).all(1)
    p=closest_on_triangles(mean,xyz[tri[take]])
    # Fallback is an actual observed point; never interpolate across unknown.
    return (xyz[peak] if p is None else p),float(logits[peak])


def compatibility_proposals(x,y,confidence,source_index,seeds=3):
    if len(x)<3:return []
    dx=np.linalg.norm(x[:,None]-x[None],axis=2);dy=np.linalg.norm(y[:,None]-y[None],axis=2)
    C=np.maximum(0,1-((dx-dy)/.03)**2);C[source_index[:,None]==source_index[None]]=0
    C=C*(C@C)/max(len(x),1);v=np.maximum(confidence,1e-5);v/=max(np.linalg.norm(v),1e-12)
    for _ in range(12):v=C@v;v/=max(np.linalg.norm(v),1e-12)
    out=[]
    for seed in np.argsort(-v)[:seeds]:
        weight=C[seed]*v;ii=np.argsort(-weight)[:min(24,len(x))]
        T=rigid(x[ii],y[ii],weight[ii]+1e-10)
        if T is not None:out.append(T)
    return out
