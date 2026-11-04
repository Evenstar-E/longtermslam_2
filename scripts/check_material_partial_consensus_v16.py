"""Mathematical correctness only; synthetic checks are NOT task evidence."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts.run_material_partial_consensus_v16 import configuration
from inspection_decisions.material_partial_consensus_graph_v16 import *
from scripts.run_textured_material_comparison_v1 import save


def main():
    torch.set_num_threads(1);torch.set_flush_denormal(True);cfg=configuration();rng=np.random.default_rng(160);x=rng.normal(0,.2,(48,3));T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();T[:3,3]=[.4,-.2,.1];y=apply(T,x);src=np.repeat(np.arange(48),3);target=np.arange(144);X=x[src];Y=np.stack([y,y+rng.normal(0,.15,y.shape),rng.normal(0,.3,y.shape)],1).reshape(-1,3);prior=np.tile([.12,.08,.05],48);m=dict(source=src,point=Y,mass=prior);a=dict(xyz=x);checks=0
    selected=source_minimum(np.sum((apply(T,X)-Y)**2,1),src);assert np.array_equal(selected,np.arange(0,144,3));checks+=1
    loss,w,r,si=partial_statistics(T,X,Y,src,target,prior,cfg,1.);assert np.isfinite(loss) and (np.bincount(si,weights=w)<=1+1e-12).all() and (np.bincount(target,weights=w)<=1+1e-12).all();checks+=1
    shared=np.zeros(144,int);_,w,_,si=partial_statistics(T,X,Y,src,shared,prior,cfg,1.);assert w.sum()<=1+1e-12 and (np.bincount(si,weights=w)<=1+1e-12).all();checks+=1
    # Original row mass, selected local mass and null sum to one for any subset.
    ix=np.arange(64);mass=np.bincount(src[ix],weights=prior[ix]);assert np.allclose(mass+np.maximum(0,1-mass),1);checks+=1
    assert tls_energy(T,X,Y,src,cfg)<1e-20;assert tls_energy(np.eye(4),X,Y,src,cfg)>.9;checks+=1
    out,tr=proposals(a,m,dict(xyz=Y),cfg)
    for f in FAMILIES:
        assert len(out[f])<=cfg['seed_budget'];R=out[f][:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
    for key,path in tr.items():
        if key.endswith('_path'):assert (np.diff(path)<=1e-10).all();checks+=1
        if key.startswith('local3_tls') and key.endswith('_initial'):assert np.array_equal(path,tr[key.replace('local3_tls','local3_partial')]);checks+=1
    assert len(out['local3_tls'])==len(out['local3_partial']);checks+=1
    # Both solvers preserve an exact clean rigid alignment; no claimed minority gain.
    g=dict(x=x,y=y);clean=dict(source=np.arange(len(x)),mass=np.full(len(x),.25));ix=np.arange(len(x))
    for partial in [False,True]:
        Q,path=refine(T.copy(),g,clean,ix,cfg,partial,1.);assert np.max(abs(Q-T))<1e-8;checks+=1
    save('results/material_partial_consensus_checks_v16.json',dict(passed=True,checks=checks,synthetic_correctness_only=True,local3_starts_identical=True,monotone_local_objectives=True));print('V16 checks',checks)


if __name__=='__main__':main()
