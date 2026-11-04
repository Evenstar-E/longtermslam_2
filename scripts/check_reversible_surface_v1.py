"""Mechanism checks, not synthetic task-performance evidence."""
import json
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.reversible_surface_v1 import root_mask,fit_rigid,transform,inverse,make_surface,costs,voxelize


def main():
    cfg=json.load(open('configs/rescan_reversible_surface_v1.json'));checks=[]
    h=np.array([[0,0,0],[.02,0,0],[.04,0,0]],float);hist={'xyz':h,'rgb':np.zeros_like(h),'tree':cKDTree(h),'class':5}
    x=np.r_[h,[[.10,0,0],[.16,0,0],[2,0,0],[2.02,0,0],[2.04,0,0]]];rgb=np.zeros_like(x)
    good,_=root_mask(x,rgb,hist,cfg);assert np.all(good[:5]) and not np.any(good[5:]);checks.append('rooted observed extension retained; disconnected mutual-support island rejected')
    bad,_=root_mask(x+10,rgb,hist,cfg);assert not bad.any();checks.append('no historical root implies no admitted increments')
    rng=np.random.default_rng(42);q=rng.normal(size=(30,3));T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];T[:3,3]=[2,3,-1]
    fit=fit_rigid(transform(q,T),q,np.ones(len(q)));assert np.max(np.abs(fit-T))<1e-12;checks.append('proper shared rigid fit and transform direction')
    assert np.allclose(inverse(transform(q,T),T),q);checks.append('fixed canonical anchor transform invertible')
    obs={'xyz':x,'rgb':rgb,'class':np.full(len(x),99)};a=np.zeros(len(x),int)
    inc,_=make_surface(obs,a,[np.eye(4)],[hist],cfg,True)
    assert np.array_equal(inc[0]['source_voxel'],np.arange(5));checks.append('source provenance preserved through filtering')
    a[:]=-1;empty,_=make_surface(obs,a,[np.eye(4)],[hist],cfg,True);assert empty[0]['tree'] is None;checks.append('ownership retraction removes entire prior increment')
    c,aa,_,_=costs(obs,[np.eye(4)],[hist],None,cfg)
    assert aa[0]==0 and aa[-1]==-1 and np.isfinite(c).all();checks.append('semantic mismatch remains candidate; unsupported geometry becomes unknown')
    un,_=make_surface(obs,np.zeros(len(x),int),[np.eye(4)],[hist],cfg,False)
    c1,_,_,_=costs(obs,[np.eye(4)],[hist],un,cfg);c2,_,_,_=costs(obs,[np.eye(4)],[hist],inc,cfg)
    assert c1[-1,0]<cfg['null_cost']<c2[-1,0];checks.append('unrooted cross-source support can explain island; root constraint changes this mechanism')
    v=voxelize(np.r_[x,x],np.zeros((2*len(x),3)),np.ones(2*len(x)),.04)
    assert v['mass'].sum()==2*len(x) and len(v['inverse'])==2*len(x);checks.append('voxel compression retains all original-point mass and inverse map')
    print(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__=='__main__':main()
