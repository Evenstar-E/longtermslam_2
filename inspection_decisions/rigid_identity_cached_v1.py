"""Exact caching of the original objective; engineering runtime repair only."""
import numpy as np
from inspection_decisions.rigid_identity_bounds_v1 import JointProblem,transform


class CachedJointProblem(JointProblem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._pair_cache={};self.cache_hits=0

    def _values(self,j,i,T):
        key=(j,i,np.ascontiguousarray(T,dtype=np.float64).tobytes())
        if key in self._pair_cache:self.cache_hits+=1;return self._pair_cache[key]
        c=self.C[j];R=T[:3,:3];t=T[:3,3]
        d=self.tree[i].query((c['xyz']-t)@R,workers=1)[0];self.point_queries+=len(d)
        p=transform(self.anchor[i][None],T)[0];norm=np.linalg.norm(p-self.anchor[i]);tau=self.cfg['distance_scale_m']
        # Retain original expression ordering for both former public paths.
        motion_full=self.cfg['motion_weight']*min(float(norm),1.)
        full=float(self.cfg['geometry_weight']*np.minimum(d/tau,1).dot(np.minimum(d/tau,1))/len(d)+self.appearance[j,i]+motion_full)
        pair=float(self.cfg['geometry_weight']*np.mean(np.minimum(d/tau,1)**2)+self.appearance[j,i]+self.cfg['motion_weight']*min(norm,1.))
        self._pair_cache[key]=(full,pair);return full,pair

    def costs(self,poses):
        return np.array([[self._values(j,i,T)[0] for i,T in enumerate(poses)] for j in range(self.n)])

    def pair_cost(self,j,i,T):return self._values(j,i,T)[1]

    def energy(self,a,poses):
        return float(sum(self.mass[j]*(self.cfg['null_cost'] if i<0 else self._values(j,i,poses[i])[0]) for j,i in enumerate(a)))
