"""Equation and ray conservation checks, not efficacy evidence."""
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from inspection_decisions.first_return_gamma_material_v1 import ray_stats,log_gamma,pair_costs,scalar_cost
from scripts.run_sparse_ray_material_incidence_v3 import save


def checks():
    c={'gamma_shape':.5,'gamma_rate_m':.08,'beta_alpha':1.,'beta_beta':1.}
    errors=[];count=0;rng=np.random.default_rng(20261025)
    # Censor at the center: no evidence behind the measured return.
    Q,meta=ray_stats(np.array([-2.,0.1,.1]),np.array([[1.,0,0]]),np.array([2.]),[False],1.,.5)
    assert Q[0].sum()==0 and abs(Q[1].sum()-1)<1e-13
    centers=(np.indices((4,4,4))[0].ravel()+.5)*.5-1
    assert Q[1,centers>0].sum()==0;count+=3
    Q,_=ray_stats(np.array([-2.,.1,.1]),np.array([[1.,0,0]]),np.array([2.1]),[True],1.,.5)
    assert Q[0].sum()==1 and abs(Q[1].sum()-1.1)<1e-13;count+=2
    # Parallel rays outside the workspace never add exposure.
    Q,_=ray_stats(np.array([-2.,2.,.1]),np.array([[1.,0,0]]),np.array([4.]),[False],1.,.5)
    assert Q.sum()==0;count+=1
    # Random segment length conservation.
    for _ in range(60):
        O=rng.uniform(-.9,.9,3);D=rng.normal(size=(1,3));D/=np.linalg.norm(D);s=rng.uniform(.001,.08)
        Q,_=ray_stats(O,D,[s],[False],1.,.125);assert abs(Q[1].sum()-s)<1e-12;count+=1
    for _ in range(30):
        a=rng.uniform(.5,4);b=rng.uniform(.5,3);n=int(rng.integers(0,5));L=rng.uniform(0,2)
        integral=quad(lambda x:x**(a+n-1)*np.exp(-(b+L)*x)*b**a/gamma(a),0,np.inf,epsabs=1e-10)[0]
        error=abs(np.log(integral)-log_gamma(a,b,n,L));assert error<1e-7;errors.append(error);count+=1
    for _ in range(15):
        H=np.stack([rng.integers(0,5,9),rng.uniform(0,2,9),rng.integers(0,10,9)]);A=rng.uniform(0,2,(4,3,9));B=rng.uniform(0,2,(3,3,9));A[:,0]=np.floor(A[:,0]);B[:,0]=np.floor(B[:,0]);A[:,:,rng.random(9)<.4]=0;B[:,:,rng.random(9)<.4]=0
        for method in ['gamma_joint','gamma_independent','gamma_mean','beta_joint']:
            C=pair_costs(H,A,B,c,method)
            for i in range(4):
                for j in range(3):assert abs(C[i,j]-scalar_cost(H,A[i],B[j],c,method))<1e-10;count+=1
    H=np.zeros((3,2));A=np.array([[[1.,0],[.1,0],[2.,0]]]);B=np.array([[[0,1.],[0,.2],[0,3.]]])
    assert abs(pair_costs(H,A,B,c,'gamma_joint')[0,0]-pair_costs(H,A,B,c,'gamma_independent')[0,0])<1e-14;count+=1
    # Integrating first-event density on finite path plus no-return atom = 1.
    for lam in [.1,1,30]:assert abs(quad(lambda s:lam*np.exp(-lam*s),0,2)[0]+np.exp(-2*lam)-1)<1e-12;count+=1
    return {'passed':True,'checks':count,'gamma_quadrature_max_log_error':max(errors),'depth_noise_integrated':False}


if __name__=='__main__':
    r=checks();save('research/first_return_gamma_checks_v1.json',r);print(r)
