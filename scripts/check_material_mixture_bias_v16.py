"""One analytic counterexample, NOT a task efficacy experiment or tuning."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import numpy as np
from scripts.run_material_partial_consensus_v16 import configuration
from inspection_decisions.material_partial_consensus_graph_v16 import partial_statistics,refine
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();x=np.array([[0.,0,0],[.3,0,0],[0,.3,0],[0,0,.3]]);source=np.repeat(np.arange(4),2);X=x[source];Y=X.copy();Y[1::2,0]+=.02;m=dict(source=source,mass=np.full(8,.25));g=dict(x=X,y=Y);T=np.eye(4);target=np.arange(8);loss,w,r,si=partial_statistics(T,X,Y,source,target,m['mass'],c,1.)
    # d(-mean log p)/dt = mean_i sum_e r_e (Tx_i-y_e)/sigma^2.
    analytic=float(np.sum(r*(X-Y)[:,0])/4/c['fit_sigma_m']**2);h=1e-6;Q=T.copy();Q[0,3]=h;P=T.copy();P[0,3]=-h;numeric=(partial_statistics(Q,X,Y,source,target,m['mass'],c,1.)[0]-partial_statistics(P,X,Y,source,target,m['mass'],c,1.)[0])/(2*h);assert abs(analytic-numeric)<1e-6
    tls,tp=refine(T.copy(),g,m,np.arange(8),c,False,1.);partial,pp=refine(T.copy(),g,m,np.arange(8),c,True,1.);assert np.max(abs(tls-T))<1e-10 and partial[0,3]>.005 and pp[-1]<pp[0]
    save('results/material_mixture_bias_check_v16.json',dict(passed=True,analytic_gradient_per_m=analytic,finite_difference_gradient_per_m=numeric,tls_translation_m=tls[:3,3].tolist(),partial_translation_m=partial[:3,3].tolist(),partial_objective_drop=float(pp[0]-pp[-1]),scope='Single noiseless synthetic counterexample: one correct and one20mm shifted mode per source, equal priors. Demonstrates possible soft-mode bias from TRUE initialization, NOT task performance, prevalence, new theorem or tuning.'))
    print('Analytic/FD gradient',analytic,numeric,'partial translation',partial[:3,3])


if __name__=='__main__':main()
