"""Measured metric patches and a small geometry/appearance descriptor.

No private labels or filesystem reads in this module. Absolute world position,
camera translation, slot numbers and acquisition groups never enter the network.
The tangent reference uses a fixed world axis; this is NOT an SE(3)-equivariant net.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from inspection_decisions.textured_material_matching_v1 import world_points,apply


def farthest_indices(points, count):
    if len(points)<=count:return np.arange(len(points))
    first=int(np.argmin(np.sum((points-points.mean(0))**2,axis=1)))
    distance=np.full(len(points),np.inf);out=[]
    for _ in range(count):
        out.append(first)
        distance=np.minimum(distance,np.sum((points-points[first])**2,axis=1))
        distance[np.array(out)]=-1.
        first=int(distance.argmax())
    return np.array(out)


def sample_frame(frame, per_slot):
    result={}
    for slot in np.unique(frame['anonymous_slot']):
        if slot<0:continue
        y,x=np.nonzero((frame['anonymous_slot']==slot)&frame['depth_valid'])
        # Bound pre-FPS cost without consulting image appearance.
        take=np.linspace(0,len(x)-1,min(len(x),1800),dtype=int)
        uv=np.c_[x[take],y[take]];xyz=world_points(frame,uv)
        ii=farthest_indices(xyz,per_slot)
        result[int(slot)]=(uv[ii].astype(float),xyz[ii])
    return result


def metric_patches(frame, uv, cfg):
    """RGB x2 + height/mask x2, sampled on locally estimated measured planes."""
    uv=np.asarray(uv,float);n=len(uv);size=cfg['patch_size']
    if not n:return np.zeros((0,10,size,size),np.float32),np.zeros((0,3))
    h,w=frame['depth'].shape
    pix=np.rint(uv).astype(int);pix[:,0]=np.clip(pix[:,0],0,w-1);pix[:,1]=np.clip(pix[:,1],0,h-1)
    xyz=world_points(frame,uv)
    slots=frame['anonymous_slot'][pix[:,1],pix[:,0]]
    r=cfg['normal_window_px'];a=np.arange(-r,r+1);dx,dy=np.meshgrid(a,a)
    nx=np.clip(pix[:,0,None]+dx.ravel(),0,w-1);ny=np.clip(pix[:,1,None]+dy.ravel(),0,h-1)
    nearuv=np.stack([nx,ny],axis=-1);near=world_points(frame,nearuv.reshape(-1,2)).reshape(n,-1,3)
    valid=(frame['anonymous_slot'][ny,nx]==slots[:,None])&frame['depth_valid'][ny,nx]
    weight=valid.astype(float);weight/=np.maximum(weight.sum(1,keepdims=True),1)
    mean=np.sum(near*weight[:,:,None],axis=1)
    delta=near-mean[:,None]
    cov=np.einsum('nki,nkj,nk->nij',delta,delta,weight)
    _,vv=np.linalg.eigh(cov);normal=vv[:,:,0]
    toward=frame['C'][:3,3]-xyz
    normal*=np.where(np.sum(normal*toward,axis=1)>=0,1.,-1.)[:,None]
    axis=np.tile([1.,0,0],(n,1));axis[abs(normal[:,0])>.9]=[0,1.,0]
    t1=axis-np.sum(axis*normal,axis=1)[:,None]*normal
    t1/=np.maximum(np.linalg.norm(t1,axis=1,keepdims=True),1e-8)
    t2=np.cross(normal,t1)
    grid=np.linspace(-1.,1.,size);gx,gy=np.meshgrid(grid,grid)
    output=[]
    inv=np.linalg.inv(frame['C']);K=frame['K']
    for radius in cfg['metric_radii_m']:
        pp=xyz[:,None,None,:]+radius*(gx[None,:,:,None]*t1[:,None,None,:]+gy[None,:,:,None]*t2[:,None,None,:])
        pc=apply(inv,pp.reshape(-1,3)).reshape(n,size,size,3)
        zz=np.maximum(pc[:,:,:,2],1e-8)
        mx=(pc[:,:,:,0]/zz*K[0,0]+K[0,2]).astype(np.float32).reshape(n*size,size)
        my=(pc[:,:,:,1]/zz*K[1,1]+K[1,2]).astype(np.float32).reshape(n*size,size)
        rgb=cv2.remap(frame['rgb'].astype(np.float32)/255.,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(n,size,size,3)
        depth=cv2.remap(frame['depth'].astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).reshape(n,size,size)
        observed=cv2.remap(frame['anonymous_slot'].astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=-1).reshape(n,size,size)
        mask=(observed==slots[:,None,None])&(depth>0)&(abs(depth-pc[:,:,:,2])<.06)&(pc[:,:,:,2]>0)
        wt=mask[:,:,:,None].astype(float);den=np.maximum(wt.sum((1,2),keepdims=True),1)
        mean=(rgb*wt).sum((1,2),keepdims=True)/den
        std=np.sqrt((((rgb-mean)*wt)**2).sum((1,2),keepdims=True)/den+.01)
        rgb=np.clip((rgb-mean)/std,-3,3)*wt
        height=np.clip((depth-pc[:,:,:,2])/radius,-1,1)*mask
        output.append(np.concatenate([rgb,height[:,:,:,None],mask[:,:,:,None]],axis=-1))
    # RGB branch six channels, geometry branch four channels.
    z=np.concatenate([output[0][:,:,:,:3],output[1][:,:,:,:3],output[0][:,:,:,3:],output[1][:,:,:,3:]],axis=-1)
    return z.transpose(0,3,1,2).astype(np.float32),xyz


class MaterialDescriptor(nn.Module):
    def __init__(self,dim=64):
        super().__init__()
        def branch(ch):
            return nn.Sequential(nn.Conv2d(ch,12,3,padding=1),nn.GroupNorm(3,12),nn.SiLU(),
                nn.Conv2d(12,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
                nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(216,32),nn.SiLU())
        self.rgb=branch(6);self.geometry=branch(4)
        self.fusion=nn.Linear(64,dim)
        self.matchability=nn.Sequential(nn.Linear(3*dim+3,64),nn.SiLU(),nn.Linear(64,1))
        self.persistence=nn.Sequential(nn.Linear(2*dim,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x):
        return F.normalize(self.fusion(torch.cat([self.rgb(x[:,:6]),self.geometry(x[:,6:])],dim=1)),dim=1)

    def context_logits(self,a,b):
        if len(b)==0:return a.new_full((len(a),),-20.)
        sim=a@b.T
        top,idx=sim.max(1)
        second=sim.topk(min(2,len(b)),dim=1).values[:,-1]
        prob=F.softmax(sim/.12,dim=1)
        entropy=-(prob*prob.clamp_min(1e-8).log()).sum(1)/max(np.log(len(b)),1.)
        context=torch.cat([a,b[idx],a*b[idx],torch.stack([top,top-second,entropy],dim=1)],dim=1)
        return self.matchability(context).squeeze(1)

    def persistence_logits(self,a,b):
        return self.persistence(torch.cat([abs(a-b),a*b],dim=1)).squeeze(1)


def training_loss(model,source,target,visible,positive,changed,target_neighbors,temperature=.12,auxiliary=.25):
    a,b=model(source),model(target)
    similarity=a@b.T/temperature
    # Near duplicate physical points are not false negatives.
    logits=similarity.masked_fill(target_neighbors,-1e4)
    ii=torch.nonzero(visible,as_tuple=True)[0]
    if len(ii):logits[ii,positive[ii]]=similarity[ii,positive[ii]]
    m=model.context_logits(a,b)
    # Visible positives AND a dustbin for occluded/outside source queries.
    extended=torch.cat([logits,(-m)[:,None]],dim=1)
    y=torch.where(visible,positive,torch.full_like(positive,len(b)))
    corr=F.cross_entropy(extended,y) if auxiliary else (F.cross_entropy(logits[ii],positive[ii]) if len(ii) else a.sum()*0.)
    match=F.binary_cross_entropy_with_logits(m,visible.float())
    if len(ii):
        persist=F.binary_cross_entropy_with_logits(model.persistence_logits(a[ii],b[positive[ii]]),(~changed[ii]).float())
    else:persist=corr*0
    total=corr+auxiliary*(match+persist)
    with torch.no_grad():
        accuracy=(similarity[ii].argmax(1)==positive[ii]).float().mean() if len(ii) else corr.new_tensor(0.)
    return total,dict(loss=float(total.detach()),correspondence=float(corr.detach()),
                      matchability=float(match.detach()),persistence=float(persist.detach()),retrieval=float(accuracy))
