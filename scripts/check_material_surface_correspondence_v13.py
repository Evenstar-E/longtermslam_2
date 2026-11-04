"""Correct circular phase sign, flat uncertainty, frame geometry and assignment."""
import numpy as np
from pathlib import Path
from inspection_decisions.material_surface_correspondence_v13 import (
    whiten_samples,phase_distribution,phase_angle,rotation_z,ChartPair,
    triple_values,optional_assignment,farthest_indices)
from scripts.prepare_material_surface_charts_v13 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();rng=np.random.default_rng(13);L=cfg['chart_angles'];rgb=rng.random((2,3,L,3));mask=np.ones((2,3,L),bool);tex,energy=whiten_samples(rgb,mask);checks=0
    assert np.max(abs(np.linalg.norm(tex.reshape(2,-1),axis=1)-1))<1e-12;checks+=1
    a=dict(xyz=np.zeros((2,3)),texture=tex,basis=np.tile(np.eye(3),(2,1,1)),fpfh=np.tile([1.,0],(2,1)),spectrum=np.tile([1.,0],(2,1)))
    b={k:v.copy() for k,v in a.items()};shift=3;b['texture']=np.roll(a['texture'],-shift,axis=2);b['basis']=np.tile(rotation_z(shift*2*np.pi/L),(2,1,1))
    corr,p,rel=phase_distribution(a,b,cfg);assert (p[np.arange(2),np.arange(2)].argmax(-1)==(-shift)%L).all();checks+=1
    for i in range(2):
        h=int(corr[i,i].argmax());R=b['basis'][i]@rotation_z(phase_angle(corr[i,i],h))@a['basis'][i].T;assert np.max(abs(R-np.eye(3)))<1e-12;checks+=1
    rc,_,_=phase_distribution(b,a,cfg);assert np.max(abs(corr-rc.transpose(1,0,2)[:,:,(-np.arange(L))%L]))<1e-12;checks+=1
    zero={k:v.copy() for k,v in a.items()};zero['texture']*=0;_,p,rel=phase_distribution(zero,zero,cfg);assert np.max(abs(p-1/L))<1e-12 and np.max(abs(rel))<1e-12;checks+=1
    index=farthest_indices(np.c_[np.arange(10),np.zeros((10,2))],5);assert len(set(index))==5 and 0 in index and 9 in index;checks+=1
    pairs,value=optional_assignment(np.array([[2.,-1],[1,3.]]));assert value==5 and len(pairs)==2;checks+=1
    ev=[]
    for _ in range(3):
        gain=rng.random((4,4));pairs,value=optional_assignment(gain);ev.append(dict(gain=gain,matched=pairs,value=value,potential=8.,free=0.))
    ind,_=triple_values(ev,cfg,False);shared,_=triple_values(ev,cfg,True);assert shared>=ind-1e-12 and ind>=0;checks+=1
    # Direct check of the target^T R source frame expression used in evidence.
    B=b['basis'];A=a['basis'];R=rotation_z(.2);frame=np.einsum('nji,jk,mkl->nmil',B,R,A).transpose(1,0,2,3)
    assert np.max(abs(frame[0,1]-B[1].T@R@A[0]))<1e-12;checks+=1
    save(Path('results/material_surface_correspondence_checks_v13.json'),dict(passed=True,checks=checks,scope='Synthetic correctness and sign/uncertainty checks only, not task benefit.'));print('V13 CHECKS',checks)


if __name__=='__main__':main()
