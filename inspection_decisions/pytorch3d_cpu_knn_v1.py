"""Exact Torch KNN API subset for CPU deployment; no compiled PyTorch3D extension."""
from collections import namedtuple
import torch

_KNN=namedtuple('_KNN',['dists','idx','knn'])

def knn_points(p1,p2,lengths1=None,lengths2=None,norm=2,K=1,version=-1,return_nn=False,return_sorted=True):
    if p1.ndim!=3 or p2.ndim!=3 or p1.shape[0]!=p2.shape[0] or p1.shape[2]!=p2.shape[2]:raise ValueError('KNN tensor shapes')
    if norm not in [1,2]:raise ValueError('norm must be 1 or 2')
    b,n,d=p1.shape;m=p2.shape[1]
    if lengths1 is None:lengths1=torch.full((b,),n,device=p1.device,dtype=torch.long)
    if lengths2 is None:lengths2=torch.full((b,),m,device=p2.device,dtype=torch.long)
    if (lengths2<1).any():raise ValueError('Empty reference cloud is outside deployed API')
    # Matrix multiplication avoids materializing BxNxMxD; same squared L2 metric.
    distance=torch.cdist(p1,p2,p=norm)
    if norm==2:distance=distance.square()
    valid2=torch.arange(m,device=p1.device)[None]<lengths2[:,None]
    distance=distance.masked_fill(~valid2[:,None,:],float('inf'))
    k=min(K,m);values,indices=distance.topk(k,dim=-1,largest=False,sorted=return_sorted)
    if K>m:
        values=torch.nn.functional.pad(values,(0,K-m));indices=torch.nn.functional.pad(indices,(0,K-m))
    valid1=torch.arange(n,device=p1.device)[None]<lengths1[:,None]
    validk=torch.arange(K,device=p1.device)[None,None,:]<lengths2[:,None,None]
    valid=valid1[:,:,None]&validk
    values=values.masked_fill(~valid,0);indices=indices.masked_fill(~valid,0)
    neighbors=knn_gather(p2,indices,lengths2).masked_fill(~valid[:,:,:,None],0) if return_nn else None
    return _KNN(values,indices,neighbors)

def knn_gather(x,idx,lengths=None):
    b,m,d=x.shape
    result=x[torch.arange(b,device=x.device)[:,None,None],idx]
    if lengths is not None:
        valid=torch.arange(idx.shape[-1],device=x.device)[None,:]<lengths[:,None]
        result=result.masked_fill(~valid[:,None,:,None],0)
    return result
