"""Matched-prior point-mean MAP control for the conditional variational model."""
import numpy as np
from inspection_decisions.shared_surface_identity_likelihood_v1 import Model as Base


class Model(Base):
    def __init__(self,H,O,pools,state,cfg,method='shared_map'):
        super().__init__(H,O,pools,state,cfg,'shared_mle');self.method='shared_map'

    def kl(self):
        d=self.mean[:,self.kh:]-self.prior[:,None,:]
        return float(.5*np.sum(d*d/self.priorvar))

    def update_surface(self):
        before=self.objective();new=self.mean.copy();counts=np.zeros((2,self.m,self.ku))
        for i in range(self.m):
            mass=np.zeros(self.ku);sums=np.zeros((self.ku,6))
            for t in [0,1]:
                r=np.exp(self.components(t,i)[:,self.kh:]-self.partition(self.L[t])[:,None])*self.O[t]['mass'][:,None]
                from inspection_decisions.reversible_surface_v2 import inverse
                y=np.c_[inverse(self.O[t]['xyz'],self.pose[t,i]),self.O[t]['rgb']]
                n=r.sum(0);counts[t,i]=n;mass+=n;sums+=r.T@y
            v=1/(1/self.priorvar+mass[:,None]/self.sigma**2)
            new[i,self.kh:]=v*(self.prior[i]/self.priorvar+sums/self.sigma**2)
        self.mean=new;self.L=[self.columns(t) for t in [0,1]];after=self.objective()
        assert after>=before-1e-5,(before,after)
        self.trace.append({'kind':'surface','before':before,'after':after});self.counts=counts
