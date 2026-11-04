from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts.run_provenance_pose_hypotheses_v1 import read
from scripts.check_provenance_pose_hypotheses_v1 import hist,obs
from inspection_decisions.shared_surface_identity_likelihood_v1 import Model


def main():
    c=read('configs/shared_surface_identity_likelihood_v1.json');x=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
    H=[hist(x)];O=[obs(x+[.2,0,0]),obs(x+[.2,0,0])];I=np.eye(4);T=I.copy();T[0,3]=.2
    pool=[[np.array([I,T])],[np.array([I,T])]];s={'poses':np.array([[I],[I]]),'k':np.zeros((2,1),int),'labels1':np.zeros(3,int),'labels2':np.zeros(3,int)}
    for method in c['methods']:
        p=Model(H,O,pool,s,c,method);start=p.objective();p.update_surface();p.pose_sweep();p.update_surface();assert p.objective()>=start-1e-5
        for t in [0,1]:
            lp=np.c_[p.L[t],np.full(len(x),p.log_out)];assert np.allclose(np.exp(lp-logsumexp(lp,axis=1)[:,None]).sum(1),1)
            assert np.allclose(p.L[t],p.columns(t))
        if method=='shared_vb':
            assert p.kl()>=-1e-8 and np.all(p.var>0)
            vb=p.objective();p.method='shared_mle';p.L=[p.columns(t) for t in [0,1]];assert p.objective()>=vb-1e-8
    # Duplicate each identical component with split weight must not create evidence.
    h=[hist(np.zeros((1,3)))];o=[obs(np.zeros((2,3))) for _ in [0,1]]
    st={**s,'labels1':np.zeros(2,int),'labels2':np.zeros(2,int)}
    a=Model(h,o,pool,st,c,'frozen_shared');cc={**c,'history_components':64,'new_components':64}
    b=Model(h,o,pool,st,cc,'frozen_shared');assert abs(a.objective()-b.objective())<1e-10
    print('PASS: normalized responsibilities, component-duplication invariance, MLE/VB coordinate improvement, full cache rebuild, positive posterior variance/KL, variational lower bound')


if __name__=='__main__':main()
