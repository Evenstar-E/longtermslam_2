"""Analytic field/SE3 derivatives, composition, marginalization, descent tests."""
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import (
    project_derivatives,persistence_factors,propagation,joint_fit)
from inspection_decisions.material_inverse_plane_control_v11 import project
from inspection_decisions.material_surface_patch_registration_v9 import inverse,retract6
from inspection_decisions.material_correspondence_mode_graph_v6 import skew
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();Y,X=np.indices((40,40));K=np.array([[25.,0,20],[0,25.,20],[0,0,1.]])
    f=dict(C=np.eye(4),K=K,depth=1/(.5+.005*(X-20)+.003*(Y-20)),depth_valid=np.ones((40,40),bool),anonymous_slot=np.zeros((40,40),int),rgb=np.stack([X/40,Y/40,(X+Y)/80],2)*255)
    # Nontrivial camera pose catches world/camera derivative transpose errors.
    f['C'][:3,:3]=Rotation.from_rotvec([.2,-.1,.05]).as_matrix();f['C'][:3,3]=[.1,.2,-.3]
    uv=np.array([[20.2,20.3],[18.3,22.1],[21.6,19.7]]);z=1/(.5+.005*(uv[:,0]-20)+.003*(uv[:,1]-20));p=np.c_[(uv[:,0]-20)*z/25,(uv[:,1]-20)*z/25,z];world=p@f['C'][:3,:3].T+f['C'][:3,3]
    r=project_derivatives(world,f,0);checks=0;maximum=0.;eps=1e-6
    for key,jkey in [('gap','gapJ'),('rgb','rgbJ')]:
        for k in range(3):
            delta=np.eye(3)[k]*eps;numerical=(project(world+delta,f,0)[key]-project(world-delta,f,0)[key])/(2*eps)
            error=float(np.max(abs(numerical-r[jkey][...,k])));assert error<1e-7,(key,error);maximum=max(maximum,error);checks+=1
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([.3,.1,-.2]).as_matrix();T[:3,3]=[.2,.1,.4]
    for direction in [0,1]:
        x=world;U=inverse(T) if direction else T;w=x@U[:3,:3].T+U[:3,3]
        J=np.concatenate([-skew(w),np.broadcast_to(np.eye(3),(len(w),3,3))],2)
        if direction:J=-np.einsum('ij,njk->nik',T[:3,:3].T,np.concatenate([-skew(x),np.broadcast_to(np.eye(3),(len(x),3,3))],2))
        for k in range(6):
            d=np.eye(6)[k]*eps;A=retract6(T,d);B=retract6(T,-d)
            if direction:A=inverse(A);B=inverse(B)
            numerical=((x@A[:3,:3].T+A[:3,3])-(x@B[:3,:3].T+B[:3,3]))/(2*eps)
            error=float(np.max(abs(numerical-J[:,:,k])));assert error<1e-8;maximum=max(maximum,error);checks+=1
    A=T;C=retract6(T,np.array([.02,0,0,0,.1,0]));B=C@A
    composed=propagation(A[None],B[None],C[None]);assert len(composed)==1 and np.max(abs(composed[0]-[A,B]))<1e-10;checks+=1
    s1=np.array([[0,1,.4]]);s2=np.array([[1,0,.7]]);value=persistence_factors(s1,s2,cfg)
    assert np.isfinite(value).all() and (value>=0).all() and np.max(abs(value-persistence_factors(s2,s1,cfg)))<1e-12;checks+=1
    value=persistence_factors(np.ones((1,4)),np.ones((1,4)),cfg);assert value[0,1]<value[0,0];checks+=1
    value=persistence_factors(np.zeros((1,4)),np.zeros((1,4)),cfg);assert np.max(abs(value-1))<1e-12;checks+=1
    # Exact brute sum over the binary state, independently written.
    pi=cfg['shared_prior'];eta=cfg['material_null_likelihood'];L=[eta+(1-eta)*s for s in [s1,s2]]
    brute=sum(prob*(L[0] if state else eta)*(L[1] if state else eta) for state,prob in [(0,1-pi),(1,pi)])
    expected=-np.log(brute).mean(-1)/(-2*np.log(eta));assert np.max(abs(expected-persistence_factors(s1,s2,cfg)[:,1]))<1e-12;checks+=1
    class Quadratic:
        def score(self,P):
            e=((P[:,:,:3,3]-np.array([.01,-.005,.002]))**2).sum((1,2));return np.tile(e[:,None],(1,4))
    U,path=joint_fit(Quadratic(),np.tile(np.eye(4),(2,1,1)),cfg,3);assert path[-1]<path[0] and np.max(np.diff(path))<=0;checks+=1
    save(Path('results/material_inverse_plane_pose_checks_v12.json'),dict(passed=True,checks=checks,max_derivative_error=maximum,meaning='Correctness only, not empirical efficacy or originality.'));print('V12 CHECKS',checks,'MAX ERROR',maximum)


if __name__=='__main__':main()
