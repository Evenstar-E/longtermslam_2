"""Finite-difference, profile-nuisance and feasible-iteration preflight."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from inspection_decisions.material_supported_continuous_alignment_v1 import (
    AXES,material_residual,material_statistics,pair_costs,projected_gn,project_parameters,
    world_pose,icp_residual)


def checks():
    cfg={'sigma_normal':.03,'sigma_offset_m':.006,'gn_rcond':1e-8,'backtracking_steps':5,'descent_tolerance':1e-10}
    rng=np.random.default_rng(66137);count=0;max_error=0.
    for _ in range(16):
        x=rng.normal(0,.03,12);features=[]
        for t in [0,1]:
            ff=[]
            for k in range(6):
                for v in range(1+(k%2)):
                    ff.append({'axis':k,'normal':AXES[k]+rng.normal(0,.001,3),'centroid':rng.normal(0,.2,3)})
            features.append(ff)
        reference={0:.24,3:.24}
        for shared in [False,True]:
            r,J=material_residual(x,features,reference,cfg,shared)
            for j in range(12):
                delta=np.eye(12)[j]*1e-6
                numeric=(material_residual(x+delta,features,reference,cfg,shared)[0]-material_residual(x-delta,features,reference,cfg,shared)[0])/2e-6
                error=float(np.max(np.abs(numeric-J[:,j])));max_error=max(max_error,error);assert error<1e-6;count+=1
            Ts=[world_pose(x[t*6:t*6+6],np.zeros(3)) for t in [0,1]]
            candidates=[[{'T':T,'slot':0}] for T in Ts];ind,joint=pair_costs(candidates,[[ff] for ff in features],[np.zeros(3)],reference,cfg)
            assert abs(float(r@r)-(joint if shared else ind)[0,0])<1e-7;count+=1
        assert joint[0,0]>=ind[0,0]-1e-8;count+=1
        fitted,log=projected_gn(x,lambda xx:material_residual(xx,features,reference,cfg,True),5,(.23,.16),cfg)
        for step in log['accepted_steps']:
            xx=np.array(step['parameters']).reshape(2,6);assert (np.linalg.norm(xx[:,:3],axis=1)<=.23+1e-12).all() and (np.linalg.norm(xx[:,3:],axis=1)<=.16+1e-12).all();count+=1
            assert step['after']<step['before'];count+=1
    # Single current-born observation has no offset/translation information;
    # two periods share a measurable difference, not a null penalty.
    f=[[{'axis':0,'normal':AXES[0],'centroid':np.array([.2,0.,0.])}],[{'axis':0,'normal':AXES[0],'centroid':np.array([.3,0.,0.])}]]
    x=np.zeros(12);ri,_=material_residual(x,f,{},cfg,False);rs,_=material_residual(x,f,{},cfg,True)
    assert ri@ri==0 and abs(rs@rs-.5*.1**2/.006**2)<1e-8;count+=1
    a,_=material_residual(x,[f[0],[]],{},cfg,True);assert a@a==0;count+=1
    # Nearest-neighbor point-to-plane derivative on a stable assignment.
    source=rng.normal(0,.1,(30,3));target=source+rng.normal(0,.002,(30,3));normal=rng.normal(size=(30,3));normal/=np.linalg.norm(normal,axis=1)[:,None];tree=cKDTree(target);x=rng.normal(0,.001,6)
    r,J=icp_residual(x,source,target,normal,tree,.2)
    for j in range(6):
        delta=np.eye(6)[j]*1e-7;numeric=(icp_residual(x+delta,source,target,normal,tree,.2)[0]-icp_residual(x-delta,source,target,normal,tree,.2)[0])/2e-7
        assert np.max(np.abs(numeric-J[:,j]))<1e-6;count+=1
    print({'status':'PASS','checks':count,'max_material_jacobian_error':max_error},flush=True)
    return {'status':'PASS','checks':count,'max_material_jacobian_error':max_error}


if __name__=='__main__':checks()
