"""Independent small exhaustive and derivative checks for the new prototype."""
import json,itertools
from pathlib import Path
import numpy as np
from inspection_decisions.material_partial_track_transport_v7 import *
from scripts.run_textured_material_comparison_v1 import save

def main():
    rng=np.random.default_rng(261026);checks=0;improvements=0
    for n in range(1,5):
        for m in range(1,5):
            for _ in range(8):
                g=rng.normal(size=(n,m));pairs,value=optional_assignment(g);brute=0.
                for a in itertools.product(range(-1,m),repeat=n):
                    used=[x for x in a if x>=0]
                    if len(set(used))==len(used):brute=max(brute,sum(g[i,j] for i,j in enumerate(a) if j>=0))
                assert abs(value-brute)<1e-10;checks+=1
    for _ in range(100):
        gains=[rng.normal(size=(4,4)) for i in range(3)];c,p=material_partition(gains,2);c0,p0=material_partition(gains,0)
        assert (np.diff(p)>=-1e-10).all() and p[-1]>=p0[-1]-1e-10
        assert p[-1]<=sum(optional_assignment(g)[1] for g in gains)+1e-10
        for e in range(3):assert np.array_equal(np.sort(c[c[:,e]>=0,e]),np.arange(4))
        improvements+=int(p[-1]>p0[-1]+1e-10);checks+=5
    assert improvements>0
    g=[np.eye(3)*2-1,np.full((3,2),-10.),np.full((3,2),-10.)];c,p=material_partition(g,2)
    assert sum((c[:,:2]>=0).all(1))==3 and sum(c[:,2]>=0)==2;checks+=1
    f=dict(C=np.eye(4),K=np.array([[10.,0,5],[0,10,5],[0,0,1]]),depth=np.ones((11,11)),depth_valid=np.ones((11,11),bool),anonymous_slot=np.zeros((11,11),int))
    e=exposure(np.array([[0,0,1],[0,0,.8],[0,0,1.2],[10,0,1.]]),[f],0,.015)
    assert e['hit'].tolist()==[True,False,False,False] and e['active'].tolist()==[False,True,False,False];checks+=1
    T=retract(np.array([np.eye(4),np.eye(4)]),rng.normal(size=12)*.1);G=np.r_[np.eye(4)[None],T];x=rng.normal(size=(6,3));error=0.
    for h in range(3):
        for f in range(3):
            if h==f:continue
            z,J=transformed_jacobian(G,h,f,x)
            for k in range(12):
                d=np.eye(12)[k]*1e-6;A=np.r_[np.eye(4)[None],retract(T,d)];B=np.r_[np.eye(4)[None],retract(T,-d)];num=(transformed_jacobian(A,h,f,x)[0]-transformed_jacobian(B,h,f,x)[0])/2e-6;error=max(error,float(np.max(abs(num-J[:,:,k]))));checks+=1
    assert error<1e-7
    p=geometry_starts(rng.normal(size=(20,3)),rng.normal(size=(30,3)));assert len(p)==24 and np.allclose(np.linalg.det(p[:,:3,:3]),1);checks+=1
    out=dict(passed=True,checks=checks,jacobian_max_error=error,random_partition_improvements=improvements,meaning='Correctness only, not task gains or novelty')
    save(Path('results/material_partial_track_transport_unit_v7.json'),out);print(json.dumps(out))

if __name__=='__main__':main()
