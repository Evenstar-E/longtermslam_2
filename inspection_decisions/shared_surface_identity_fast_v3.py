"""Same bounded kernel, avoid constructing far-point component tensors per pose."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from inspection_decisions.shared_surface_identity_fast_v2 import Model as Fast
from inspection_decisions.shared_surface_identity_likelihood_v1 import Model as Dense
from inspection_decisions.shared_surface_identity_map_control_v1 import Model as MAP
from inspection_decisions.reversible_surface_v2 import transform


class Model(Fast):
    def asset_column(self,t,i,T=None,obs=None):
        o=self.O[t] if obs is None else obs;T=self.pose[t,i] if T is None else T
        x=self.mean[i,:,:3];center=(x.min(0)+x.max(0))/2
        radius=np.linalg.norm(x-center,axis=1).max()+self.tail_sigma*self.cfg['position_sigma_m']+1e-8
        center=transform(center[None],T)[0]
        if obs is None:
            if not hasattr(self,'spatial_trees'):self.spatial_trees=[cKDTree(q['xyz']) for q in self.O]
            idx=np.array(sorted(self.spatial_trees[t].query_ball_point(center,radius)),int)
        else:idx=np.flatnonzero(np.linalg.norm(o['xyz']-center,axis=1)<=radius)
        col=np.full(len(o['xyz']),-1e300)
        if len(idx):col[idx]=logsumexp(Dense.components(self,t,i,T,{k:o[k][idx] for k in ['xyz','rgb','class']}),axis=1)
        return col

    def columns(self,t,obs=None):return np.column_stack([self.asset_column(t,i,obs=obs) for i in range(self.m)])

    def pose_sweep(self,monitor=None):
        for t in [0,1]:
            for i in range(self.m):
                before=self.objective();others=self.L[t].copy();others[:,i]=-np.inf
                other=self.partition(others);best=np.dot(self.O[t]['mass'],np.logaddexp(other,self.L[t][:,i]));chosen=None;records=[]
                for k,T in enumerate(self.pool[t][i]):
                    col=self.asset_column(t,i,T);value=float(np.dot(self.O[t]['mass'],np.logaddexp(other,col)));records.append(value)
                    if value>best+1e-7:best=value;chosen=(k,T,col)
                if chosen:
                    k,T,col=chosen;self.k[t,i]=k;self.pose[t,i]=T;self.L[t][:,i]=col
                after=self.objective();assert after>=before-1e-5
                row={'kind':'pose','period':t+1,'asset_uid':self.H[i]['uid'],'asset_index':i,'before':before,'after':after,'accepted':chosen is not None,'candidate_values':records,'selected_k':int(self.k[t,i])}
                self.trace.append(row)
                if monitor:monitor(row)


class MAPModel(MAP,Model):
    pass
