"""Ordinary multi-period registration of independently matched pair edges.

This is a standard strong control, not a persistent-material algorithm or BA
with material landmarks. Unlike V6 it does not require three observed edges.
"""
import numpy as np
from inspection_decisions.textured_material_matching_v1 import apply
from inspection_decisions.material_correspondence_inference_v3 import inverse
from inspection_decisions.material_correspondence_mode_graph_v6 import skew,retract


def residual_jacobian(T,edges):
    rr=[];JJ=[];ww=[];G1,G2=T;R=G2[:3,:3]@G1[:3,:3].T
    for pe,r in enumerate(edges):
        x,y=r['x'],r['y'];N=len(x);J=np.zeros((N,3,12))
        if pe==0:
            z=apply(G1,x);J[:,:,:3]=-skew(z);J[:,:,3:6]=np.eye(3)
        elif pe==1:
            z=apply(G2,x);J[:,:,6:9]=-skew(z);J[:,:,9:12]=np.eye(3)
        else:
            z=apply(G2@inverse(G1),x);J[:,:,:3]=R@skew(x);J[:,:,3:6]=-R;J[:,:,6:9]=-skew(z);J[:,:,9:12]=np.eye(3)
        w=r['weight'].copy()
        if len(w):
            # Two pairwise alternatives per node remain explicit; confidence
            # is a robust fitting weight, never called calibrated probability.
            _,a,ca=np.unique(r['source'],return_inverse=True,return_counts=True);_,b,cb=np.unique(r['target'],return_inverse=True,return_counts=True);w/=np.maximum(ca[a],cb[b]);w/=max(w.sum(),1e-12)
        rr.append(z-y);JJ.append(J);ww.append(w*(.5 if pe==2 else 1.))
    return np.concatenate(rr),np.concatenate(JJ),np.concatenate(ww)


def fit(initial,edges,steps=4,scale=.025):
    T=np.array(initial);path=[]
    def value(T):
        r,_,w=residual_jacobian(T,edges);return float(np.sum(w*scale**2*np.log1p(np.sum(r*r,axis=1)/scale**2)))
    current=value(T);path.append(current)
    for _ in range(steps):
        r,J,w=residual_jacobian(T,edges);factor=np.sqrt(w/(1+np.sum(r*r,axis=1)/scale**2));A=(J*factor[:,None,None]).reshape(-1,12);b=(r*factor[:,None]).ravel()
        if not len(b):break
        d=np.linalg.lstsq(np.r_[A,np.eye(12)*.001],np.r_[-b,np.zeros(12)],rcond=None)[0];d*=min(1,.05/max(np.linalg.norm(d),1e-12));accepted=False
        for alpha in [1.,.5,.25,.125]:
            candidate=retract(T,d*alpha);v=value(candidate)
            if v<current-1e-14:T=candidate;current=v;accepted=True;break
        path.append(current)
        if not accepted:break
    return T,np.array(path)
