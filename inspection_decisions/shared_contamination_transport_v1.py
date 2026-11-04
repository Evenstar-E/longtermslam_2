"""Shared random-effects contamination + partial capacity, discrete objective bounds."""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from inspection_decisions.contrastive_material_transport_v1 import witnesses
from inspection_decisions.ray_material_evidence_v1 import ray_states

def patches(frame,n):
    seeds,_=witnesses(frame,n);labels=cdist(frame.all_xyz,seeds).argmin(1);xyz=[];rgb=[];radius=[]
    uv=frame.all_xyz@frame.K.T;uv=np.rint(uv[:,:2]/uv[:,2,None]).astype(int);colors=frame.image[uv[:,1],uv[:,0]]
    for j in range(n):
        pts=frame.all_xyz[labels==j];center=pts.mean(0);xyz.append(center);rgb.append(colors[labels==j].mean(0));radius.append(np.mean(np.sum((pts-center)**2,axis=1))/3)
    return np.array(xyz),np.array(rgb),np.array(radius)

def assignment(w):
    i,j=linear_sum_assignment(-np.maximum(w,0));keep=w[i,j]>0;return i[keep],j[keep]

def shared_bounds(weights,ray,prior):
    n=weights.shape[1];choices=[assignment(w) for w in weights];choices.append((np.array([],int),np.array([],int)))
    upper=logsumexp(np.log(prior)+np.array([weights[k][ij].sum()+ray[k] for k,ij in enumerate(choices[:-1])]))/n
    lower=max(logsumexp(np.log(prior)+np.array([w[ij].sum() for w in weights])+ray)/n for ij in choices)
    return float(lower),float(upper)

class Model:
    def __init__(self,h,c,cfg):
        self.h=h;self.c=c;self.cfg=cfg;self.x,self.a,ha=patches(h,cfg['points']);self.y,self.b,ca=patches(c,cfg['points']);self.eps=np.array(cfg['epsilon_grid']);self.prior=np.array(cfg['epsilon_prior'])
        self.var=cfg['sensor_sigma_m']**2+ha[:,None]+ca[None,:];self.color_d2=cdist(self.a,self.b,'sqeuclidean')
        # Textureless limit is exactly geometry-only; this is a fixed reliability heuristic.
        v=float(np.var(self.a,axis=0).mean());self.gamma=v/(v+cfg['color_sigma']**2)

    def score(self,T):
        cfg=self.cfg;n=len(self.x);R=T[:3,:3];t=T[:3,3];tx=self.x@R.T+t;iy=(self.y-t)@R
        hs,*_=ray_states(iy,self.h.depth,self.h.mask,self.h.K,.015,[.001,5]);cs,*_=ray_states(tx,self.c.depth,self.c.mask,self.c.K,.015,[.001,5]);surface=((hs==1).sum()+(cs==1).sum())/2;free=((hs==2).sum()+(cs==2).sum())/2
        d2=cdist(tx,self.y,'sqeuclidean');null=cfg['spatial_null_sigma_m']**2;lg=1.5*np.log(null/self.var)-.5*d2*(1/self.var-1/null)
        colvar=cfg['color_sigma']**2;colnull=cfg['color_null_sigma']**2;lc=1.5*np.log(colnull/colvar)-.5*self.color_d2*(1/colvar-1/colnull)
        texture=np.logaddexp(np.log(max(1-self.gamma,1e-300)),np.log(max(self.gamma,1e-300))+lc) if self.gamma>1e-14 else np.zeros_like(lc)
        ray=surface*np.log(2*(1-self.eps))+free*np.log(2*self.eps);scores={};bounds={}
        for label,likelihood in [('geometry',lg),('geometry_texture',lg+texture)]:
            w=np.array([np.logaddexp(np.log(e),np.log1p(-e)+likelihood) for e in self.eps])
            independent=logsumexp(np.log(self.prior)[:,None,None]+w,axis=0);ij=assignment(independent)
            independent_ray=surface*np.log(2*(1-np.dot(self.prior,self.eps)))+free*np.log(2*np.dot(self.prior,self.eps))
            scores[label+'_independent']=float((independent[ij].sum()+independent_ray)/n)
            lb,ub=shared_bounds(w,ray,self.prior);scores[label+'_shared']=lb;bounds[label]=dict(lower=lb,upper=ub,gap=ub-lb)
            assert ub+1e-10>=lb
        return scores,dict(bounds=bounds,texture_reliability=self.gamma,known_surface=float(surface/n),known_conflict=float(free/n))
