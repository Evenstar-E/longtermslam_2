"""Independent integrals, gradients and cause accounting; no task truth."""
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from inspection_decisions.surface_measure_first_return_v1 import (surface_terms,surface_tensor,grid_exposure_tensor,unknown_exposure,causes,cost,pair_costs,PoseObjective)
from scripts.run_rigid_identity_bounds_v1 import read
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(20261025)
    cfg={**read('results/first_return_gamma_material_v5/config.json'),**read('configs/surface_measure_first_return_v1.json'),'hazard_extent_m':.32}
    q=8;model=dict(V=q**3,q=q,center=np.array([[.03,-.02,.01],[-.04,.05,.02]]),normal=np.array([[0.,0,1.],[1.,0,0.]]))
    model['precision']=np.array([np.eye(3)/.08**2+np.outer(n,n)*(1/.006**2-1/.08**2) for n in model['normal']]);model['surface_mass']=np.array([3.,3.])
    checks=0;integral_error=0.;backend_error=0.;grad_error=0.;pair_error=0.
    for repeat in range(16):
        O=rng.uniform(-.2,.2,(7,3));D=rng.normal(size=(7,3));D/=np.linalg.norm(D,axis=1)[:,None];s=rng.uniform(.1,.4,7);w=rng.uniform(.5,2,7)
        P=O+D*s[:,None];local=dict(O=O,D=D,s=s,weight=w,P=P,U=D)
        K,A=surface_terms(local,model,cfg);Kt,At=surface_tensor({k:torch.tensor(v,dtype=torch.float64) for k,v in local.items()},model,cfg)
        error=max(float(np.max(abs(K-Kt.numpy()))),float(np.max(abs(A-At.numpy()))));backend_error=max(error,backend_error);assert error<1e-8;checks+=1
        for f in range(2):
            integral=0.
            for o,d,length,weight in zip(O,D,s,w):
                n=model['normal'][f];Q=model['precision'][f];c=model['center'][f];scale=np.sqrt((n@d)**2+cfg['incidence_floor']**2)/(np.sqrt(2*np.pi)*cfg['surface_epsilon_m'])
                fun=lambda t:scale*np.exp(-.5*(o+d*t-c)@Q@(o+d*t-c))
                value=quad(fun,0,length,epsabs=1e-11,points=[float(np.clip(-d@Q@(o-c)/(d@Q@d),0,length))])[0];integral+=weight*value
            error=abs(integral-A[f]);integral_error=max(error,integral_error);assert error<1e-8;checks+=1
        # Truncation at the first observed return, and additivity of disjoint segments.
        half=dict(local,s=s/2,P=O+D*s[:,None]/2)
        rest=dict(local,O=O+D*s[:,None]/2,s=s/2)
        _,a1=surface_terms(half,model,cfg);_,a2=surface_terms(rest,model,cfg)
        assert np.max(abs(A-a1-a2))<1e-8;checks+=1
        # Full pose gradient incl. exposure and fractional-count/entropy derivatives.
        O=rng.uniform(-.1,.1,(12,3));D=rng.normal(size=(12,3));D/=np.linalg.norm(D,axis=1)[:,None];s=np.full(12,.11)
        data=dict(O=O,D=D,s=s,weight=np.ones(12),PO=O,PD=D,ps=s,owner=np.zeros(12,int))
        a=rng.uniform(.3,2,q**3+2);b=rng.uniform(.6,2,q**3+2);x=rng.uniform(-.008,.008,12)
        objective=PoseObjective([data,data],[0,0],np.zeros((1,3)),model,a,b,cfg,'surface_shared')
        value,g=objective(x);eps=2e-7
        for j in range(12):
            xp=x.copy();xm=x.copy();xp[j]+=eps;xm[j]-=eps
            numeric=(objective(xp,False)-objective(xm,False))/(2*eps);error=abs(numeric-g[j])/max(1,abs(numeric),abs(g[j]));grad_error=max(error,grad_error)
            assert error<5e-5,(repeat,j,g[j],numeric,error);checks+=1
        localO=(O-x[3:6])@Rotation.from_rotvec(x[:3]).as_matrix();localD=D@Rotation.from_rotvec(x[:3]).as_matrix()
        gt=grid_exposure_tensor(torch.tensor(localO),torch.tensor(localD),torch.tensor(s),torch.ones(12),model,cfg).numpy()
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();T[:3,3]=x[3:6]
        gn=unknown_exposure(data,T,cfg);assert np.max(abs(gt-gn))<1e-10;checks+=1
    # Pair matrix must equal the scalar formula, for REAL (fractional) counts.
    a=rng.uniform(.3,2,9);b=rng.uniform(.5,2,9)
    make=lambda:dict(n=rng.uniform(0,3,9),A=rng.uniform(0,2,9),c=float(rng.normal()),mean=float(rng.normal()),positive=float(rng.normal()))
    A=[make() for _ in range(4)];B=[make() for _ in range(5)];mat=pair_costs(a,b,A,B,cfg['methods'])
    for m in cfg['methods']:
        for i in range(4):
            for j in range(5):
                error=abs(mat[m][i,j]-cost(a,b,A[i],B[j],m));pair_error=max(pair_error,error);assert error<1e-10;checks+=1
    result=dict(passed=True,checks=checks,max_surface_integral_error=integral_error,max_numpy_torch_error=backend_error,max_pose_gradient_relative_error=grad_error,max_pair_scalar_error=pair_error,GT_read=False,full_192_completed=False)
    save('research/surface_measure_first_return_checks_v1.json',result);print(result)


if __name__=='__main__':main()
