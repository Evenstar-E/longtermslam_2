"""Correctness of proposed factor only; no task efficacy assertion."""
import numpy as np
from inspection_decisions.surface_measure_evidence_prototype_v1 import marked_evidence,partition_evidence
from scripts.run_rigid_identity_bounds_v1 import sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    rng=np.random.default_rng(20261025);eq=0.;split=0.;gradient=0.;checks=0
    for repeat in range(80):
        N=repeat%6;F=1+repeat%3;K=rng.uniform(.02,3,(N,F));A=rng.uniform(.1,3,F);mass=rng.uniform(.05,2,F);rate=.8;v,mean=marked_evidence(K,A,mass,rate);other=partition_evidence(K,A,mass,rate);eq=max(eq,abs(v-other));assert abs(v-other)<1e-10;checks+=1
        kk=np.c_[K[:,0],K];aa=np.r_[A[0],A];mm=np.r_[mass[0]/3,mass];mm[1]=mass[0]*2/3;duplicated,_=marked_evidence(kk,aa,mm,rate);split=max(split,abs(v-duplicated));assert abs(v-duplicated)<1e-10;checks+=1
        for j in range(F):
            plus=A.copy();minus=A.copy();plus[j]+=1e-5;minus[j]-=1e-5;numeric=(partition_evidence(K,plus,mass,rate)-partition_evidence(K,minus,mass,rate))/(2e-5);error=abs(numeric+mean[j]);gradient=max(gradient,error);assert error<1e-7;checks+=1
    result={'passed':True,'checks':checks,'max_mark_partition_log_evidence_error':eq,'max_duplicate_kernel_mass_split_error':split,'max_exposure_gradient_error':gradient,'GT_read':False,'full_192_surface_model_implemented':False,'prototype_source_sha256':sha('inspection_decisions/surface_measure_evidence_prototype_v1.py')};save('research/surface_measure_evidence_prototype_checks_v1.json',result);print(result)


if __name__=='__main__':main()
