import numpy as np
from scipy.sparse import csr_matrix
from inspection_decisions.uncertain_face_incidence_schur_v2 import schur_step


def checks():
    rng=np.random.default_rng(1025);maximum=0.
    for _ in range(40):
        m=50;k=9;n=12;A=rng.normal(size=(m,k));D=np.zeros((m,n));D[np.arange(m),rng.integers(0,n,m)]=rng.normal(size=m);J=np.c_[A,D];r=rng.normal(size=m);damping=.01;step,_=schur_step(csr_matrix(J),r,k,damping);reference=np.linalg.lstsq(np.r_[J,damping*np.eye(k+n)],np.r_[-r,np.zeros(k+n)],rcond=None)[0];err=float(abs(step-reference).max());maximum=max(maximum,err);assert err<1e-8
        active=rng.random(n)<.3;free=np.r_[np.ones(k,bool),~active];step,_=schur_step(csr_matrix(J),r,k,damping,active);expected=np.zeros(k+n);expected[free]=np.linalg.lstsq(np.r_[J[:,free],damping*np.eye(free.sum())],np.r_[-r,np.zeros(free.sum())],rcond=None)[0];err=float(abs(step-expected).max());maximum=max(maximum,err);assert err<1e-8
    return {'checks':80,'max_step_difference_vs_dense_damped_LS':maximum,'GT_read':False,'standard_Schur_not_novelty':True}


if __name__=='__main__':print(checks())
