"""Check the implemented finite variational term against exact cause summation.

Auxiliary check only: cannot change frozen inference or assert task gains.
"""
import numpy as np
from scipy.special import digamma,logsumexp
from inspection_decisions.surface_measure_evidence_prototype_v1 import marked_evidence
from inspection_decisions.surface_measure_first_return_v1 import causes,gamma_integral
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    rng=np.random.default_rng(20261026);checks=0;gaps=[];optimized=[]
    for repeat in range(60):
        N=repeat%6;F=1+repeat%3;rate=.8
        mass=rng.uniform(.2,2,F+1);A=rng.uniform(0,3,F+1)
        K=np.c_[np.full(N,12.5),rng.uniform(.02,12,(N,F))]
        exact,_=marked_evidence(K,A,mass,rate)
        model={'V':1};cfg={'hazard_voxel_m':.08};ids=np.zeros(N,int)
        n,c=causes(np.log(K[:,1:]),ids,mass,np.full(F+1,rate),model,cfg)
        bound=c+gamma_integral(mass,np.full(F+1,rate),n,A).sum()
        assert bound<=exact+1e-10;checks+=1;gaps.append(float(exact-bound))
        # A different approximation: joint coordinate ascent at fixed geometry.
        a=mass.copy();b=np.full(F+1,rate)+A;trace=[]
        for _ in range(50):
            logits=np.log(K)+digamma(a)[None]-np.log(b)[None];norm=logsumexp(logits,1);R=np.exp(logits-norm[:,None]);n=R.sum(0)
            c=float(norm.sum()-n@(digamma(a)-np.log(b)));a=mass+n
            value=float(c+gamma_integral(mass,np.full(F+1,rate),n,A).sum());trace.append(value)
        assert np.min(np.diff(trace),initial=0)>-1e-10 and trace[-1]<=exact+1e-10;checks+=2
        optimized.append(float(exact-trace[-1]))
    result=dict(passed=True,checks=checks,frozen_proposal_ELBO_gap_max=max(gaps),joint_coordinate_ELBO_gap_max=max(optimized),
                source_sha256=sha('inspection_decisions/surface_measure_first_return_v1.py'),scope='60 tiny no-GT synthetic factors, not task results. Frozen-proposal approximation can be looser than joint coordinate optimization. Difference of H/current ELBOs is not an exact conditional evidence bound.')
    save('research/surface_measure_variational_bound_checks_v1.json',result);print(result)


if __name__=='__main__':main()
