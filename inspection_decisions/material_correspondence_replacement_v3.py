"""Posthoc structural control: replace old texture cue instead of adding it."""
import numpy as np
import torch
from inspection_decisions.material_correspondence_inference_v3 import PairScore


class ReplacementScore(PairScore):
    def texture_numpy(self,Ts):
        mm=self.match
        if not len(mm['ia']):return np.ones(len(Ts))
        a=self.a['xyz'][mm['ia']];b=self.b['xyz'][mm['ib']]
        x=np.einsum('bij,nj->bni',Ts[:,:3,:3],a)+Ts[:,None,:3,3]
        mass=np.sum(np.exp(-.5*np.sum((x-b[None])**2,axis=2)/.025**2)*np.exp(-.5*(mm['distance']/.5)**2)[None],axis=1)
        return 1-np.minimum(1,mass/6)

    def score(self,Ts,methods=None,return_parts=False):
        methods=['ordinary','ray','hard2','soft2','soft3'] if methods is None else methods
        values=super().score(Ts,methods,return_parts=False);texture=self.texture_numpy(np.asarray(Ts))
        return {m:v if m=='ordinary' else v-texture for m,v in values.items()}

    def differentiable(self,T,method,reference):
        value=super().differentiable(T,method,reference)
        if method=='ordinary':return value
        mm=self.match
        if not len(mm['ia']):return value-1
        a=torch.as_tensor(self.a['xyz'][mm['ia']],dtype=T.dtype);b=torch.as_tensor(self.b['xyz'][mm['ib']],dtype=T.dtype)
        d=a@T[:3,:3].T+T[:3,3]-b
        mass=(torch.exp(-.5*(d*d).sum(1)/.025**2)*torch.as_tensor(np.exp(-.5*(mm['distance']/.5)**2),dtype=T.dtype)).sum()
        return value-(1-(mass/6).clamp(max=1))
