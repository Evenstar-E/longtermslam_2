"""Bounded Gaussian-tail acceleration; every identity/pose is still tested."""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.shared_surface_identity_likelihood_v1 import Model as Dense
from inspection_decisions.shared_surface_identity_map_control_v1 import Model as MAP
from inspection_decisions.reversible_surface_v2 import transform


class Model(Dense):
    tail_sigma=12.

    def components(self,t,i,T=None,obs=None):
        o=self.O[t] if obs is None else obs;T=self.pose[t,i] if T is None else T
        x=self.mean[i,:,:3];center=(x.min(0)+x.max(0))/2
        radius=np.linalg.norm(x-center,axis=1).max()+self.tail_sigma*self.cfg['position_sigma_m']+1e-8
        center=transform(center[None],T)[0]
        if obs is None:
            if not hasattr(self,'spatial_trees'):self.spatial_trees=[cKDTree(q['xyz']) for q in self.O]
            idx=np.array(sorted(self.spatial_trees[t].query_ball_point(center,radius)),int)
        else:idx=np.flatnonzero(np.linalg.norm(o['xyz']-center,axis=1)<=radius)
        result=np.full((len(o['xyz']),self.kh+self.ku),-1e300)
        if len(idx):
            sub={k:o[k][idx] for k in ['xyz','rgb','class']}
            result[idx]=Dense.components(self,t,i,T,sub)
        return result

    def tail_bound(self):
        peak=(2*np.pi)**-3/np.prod(self.sigma)
        return float(sum(o['mass'].sum() for o in self.O)*(1-self.cfg['outlier_mass'])*peak*np.exp(-.5*self.tail_sigma**2-self.log_out))


class MAPModel(MAP,Model):
    pass
