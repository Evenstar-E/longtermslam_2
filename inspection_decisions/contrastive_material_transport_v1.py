"""Capacity-constrained excess material evidence over a fixed permutation null.

Research hypothesis, not a novelty or calibrated likelihood-ratio claim.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from inspection_decisions.ray_material_evidence_v1 import ray_states

def witnesses(frame,n):
    x=frame.all_xyz;chosen=[int(np.argmin(np.linalg.norm(x-x.mean(0),axis=1)))];d=np.full(len(x),np.inf)
    for _ in range(1,n):
        d=np.minimum(d,np.sum((x-x[chosen[-1]])**2,axis=1));chosen.append(int(np.argmax(d)))
    q=x[chosen];uv=q@frame.K.T;uv=np.rint(uv[:,:2]/uv[:,2,None]).astype(int);rgb=frame.image[uv[:,1],uv[:,0]]
    return q,rgb

def evidence(w,capacity):
    # Zero reward = no claimed historical correspondence; no forced overlap.
    w=np.maximum(w,0)
    if capacity:
        i,j=linear_sum_assignment(-w);return float(w[i,j].sum()/len(w))
    return float((w.max(0).mean()+w.max(1).mean())/2)

class Model:
    def __init__(self,h,c,cfg):
        self.h=h;self.c=c;self.cfg=cfg;self.x,self.a=witnesses(h,cfg['points']);self.y,self.b=witnesses(c,cfg['points'])
        rng=np.random.default_rng(cfg['seed']);self.perms=[rng.permutation(len(self.b)) for _ in range(cfg['permutations'])]
        self.color=[cdist(self.a,b,metric='cityblock')/(3*cfg['color_scale']) for b in [self.b]+[self.b[p] for p in self.perms]]

    def score(self,T):
        R=T[:3,:3];t=T[:3,3];tx=self.x@R.T+t;iy=(self.y-t)@R
        hs,*_=ray_states(iy,self.h.depth,self.h.mask,self.h.K,.015,[.001,5]);cs,*_=ray_states(tx,self.c.depth,self.c.mask,self.c.K,.015,[.001,5])
        feasible=(cs[:,None]==1)&(hs[None,:]==1);g=(cdist(tx,self.y)/self.cfg['distance_scale_m'])**2
        free=((hs==2).mean()+(cs==2).mean())/2;values={};diagnosis={}
        for capacity,label in [(False,'independent'),(True,'capacity')]:
            scores=[evidence(np.where(feasible,1-g-color**2,0),capacity) for color in self.color]
            null=float(np.mean(scores[1:]));values[label+'_raw']=float(free-scores[0]);values[label+'_contrastive']=float(free-scores[0]+null)
            diagnosis[label]=dict(actual=scores[0],null_mean=null,excess=scores[0]-null,null_scores=scores[1:])
        return values,dict(free_fraction=float(free),evidence=diagnosis)
