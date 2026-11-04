"""Independent value/gradient agreement of the actual SE3 inference objective."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.linalg import expm
from inspection_decisions.material_correspondence_inference_v3 import PairScore,refine_joint,inverse


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(21);cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text())
    x=rng.normal(size=(20,3))*.06+[0,0,2];y=x+rng.normal(size=x.shape)*.002
    a=dict(points=x,tree=cKDTree(x),xyz=x,area=np.ones(20)/20,sigma=np.full(20,.02))
    b=dict(points=y,tree=cKDTree(y),xyz=y,area=np.ones(20)/20,sigma=np.full(20,.02))
    q=np.eye(20)*.9+.1/20;item=dict(q=q,coverage=np.full(20,.7),a=a,b=b)
    mm=dict(ia=np.arange(20),ib=np.arange(20),distance=np.ones(20)*.1)
    pair=PairScore(a,b,mm,a,b,{n:item for n in ['hard2','soft2','soft3']},[],0,cfg)
    T=np.eye(4);T[:3,3]=[.004,-.003,.001]
    actual=pair.score(T[None],['ordinary','soft3_no_visibility'])
    def pose(v):
        A=np.zeros((4,4));A[:3,:3]=[[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]];A[:3,3]=v[3:]
        return expm(A)@T
    maxerr=0.
    for method in ['ordinary','soft3_no_visibility']:
        v=torch.zeros(6,dtype=torch.double,requires_grad=True);A=torch.zeros(4,4,dtype=torch.double)
        A[0,1]=-v[2];A[0,2]=v[1];A[1,0]=v[2];A[1,2]=-v[0];A[2,0]=-v[1];A[2,1]=v[0];A[:3,3]=v[3:]
        tt=torch.matrix_exp(A)@torch.as_tensor(T,dtype=torch.double);loss=pair.differentiable(tt,method,T);loss.backward()
        assert abs(float(loss.detach())-actual[method][0])<1e-10
        fd=[]
        for k in range(6):
            d=np.zeros(6);d[k]=1e-6
            fd.append((pair.score(pose(d)[None],[method])[method][0]-pair.score(pose(-d)[None],[method])[method][0])/2e-6)
        err=np.max(abs(np.array(fd)-v.grad.numpy()));maxerr=max(maxerr,err)
        assert err<2e-4,(method,err)
    result,path=refine_joint(T,T,[pair,pair,pair],'soft3_no_visibility',cfg)
    assert np.max(np.diff(path))<=1e-10 and np.isfinite(result).all()
    assert np.max(abs(inverse(T)@T-np.eye(4)))<1e-12
    print('PASS actual NumPy/Torch objective, 12 finite differences, monotone joint SE3; maxgradient',maxerr,flush=True)


if __name__=='__main__':main()
