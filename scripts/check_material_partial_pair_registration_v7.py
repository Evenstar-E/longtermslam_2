import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.textured_material_matching_v1 import apply
from inspection_decisions.material_correspondence_mode_graph_v6 import retract
from inspection_decisions.material_partial_pair_registration_v7 import residual_jacobian,fit
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    rng=np.random.default_rng(261127);T=np.tile(np.eye(4),(2,1,1));T[0,:3,:3]=Rotation.from_rotvec([.1,-.05,.08]).as_matrix();T[0,:3,3]=[.4,-.2,.1];T[1,:3,:3]=Rotation.from_rotvec([-.05,.1,-.1]).as_matrix();T[1,:3,3]=[-.2,.3,.15]
    # DISJOINT material sets on three edges: there is no material triangle.
    edges=[]
    for pe,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
        q=rng.uniform(-.2,.2,(14+pe,3))+[1,2,0];A=np.eye(4) if a==0 else T[a-1];B=T[b-1];x=apply(A,q);y=apply(B,q)+rng.normal(0,.001,q.shape)
        edges.append(dict(x=x,y=y,source=np.arange(len(q))+pe*100,target=np.arange(len(q))+1000+pe*100,weight=np.ones(len(q))))
    r,J,w=residual_jacobian(T,edges);maximum=0.
    for k in range(12):
        d=np.zeros(12);d[k]=1e-6;fd=(residual_jacobian(retract(T,d),edges)[0]-residual_jacobian(retract(T,-d),edges)[0])/2e-6;maximum=max(maximum,float(np.max(abs(fd-J[:,:,k]))))
    assert maximum<1e-7
    initial=retract(T,rng.normal(0,.006,12));result,path=fit(initial,edges);assert np.max(np.diff(path))<=1e-14 and path[-1]<path[0] and np.linalg.norm(result-T)<np.linalg.norm(initial-T)
    no12=[dict(r) for r in edges];no12[2]={k:v[:0] for k,v in no12[2].items()};result2,path2=fit(initial,no12);assert path2[-1]<path2[0]
    dest=Path('results/material_partial_pair_registration_check_v7.json');save(dest,dict(complete=True,max_jacobian_error=maximum,columns=12,disjoint_material_sets=True,actual_three_cycle_required=False,descent_with_missing_C12=True,source_hashes={p:sha(p) for p in [__file__,'inspection_decisions/material_partial_pair_registration_v7.py']}));print('V7 STANDARD PARTIAL PAIR CHECK PASS',maximum,flush=True)


if __name__=='__main__':main()
