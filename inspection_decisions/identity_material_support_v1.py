"""Separate retained identity A from material-source admission M.

Finite deterministic prototype, not a novel probabilistic theorem or posterior.
"""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem as Base
from inspection_decisions.reversible_surface_v2 import inverse


class Problem(Base):
    def __init__(self,O,H,pools,labels,cfg,admission):
        self.M=[np.ones(len(o['xyz']),bool) for o in O]
        self.admission=admission;self.trial=None
        super().__init__(O,H,pools,labels,cfg,'pool_coupled')

    def surface(self,t,i,labels=None,poses=None):
        labels=self.z[t] if labels is None else labels
        poses=self.pose[t] if poses is None else poses
        mask=self.trial[1] if self.trial is not None and self.trial[0]==t else self.M[t]
        idx=np.flatnonzero((labels==i)&mask)
        q=inverse(self.O[t]['xyz'][idx],poses[i])
        return {'xyz':q,'rgb':self.O[t]['rgb'][idx],'source_voxel':idx,'tree':cKDTree(q) if len(q) else None}

    def propose(self,t,i,col):
        region=np.flatnonzero((self.z[t]==i)|(self.C[t][:,i]<self.null)|(col<self.null))
        C=self.C[t][region].copy();C[:,i]=col[region]
        a=np.argmin(C,axis=1);value=C[np.arange(len(region)),a]
        a[value>=self.null]=-1
        old=self.z[t][region];oldvalue=np.full(len(region),self.null)
        ok=old>=0;oldvalue[ok]=C[np.flatnonzero(ok),old[ok]]
        keep=oldvalue<=value+1e-9;a[keep]=old[keep];value[keep]=oldvalue[keep]
        z=self.z[t].copy();z[region]=a
        M=self.M[t].copy()
        if self.admission:M[region]=(a>=0)&(value<self.null)
        own=float(np.dot(self.weights[t][region],value-self.assigned(t)[region]))
        return z,M,own

    def back(self,t,i,T,z,M):
        changed=(z!=self.z[t])|(M!=self.M[t])
        affected=sorted(({i}|set(z[changed])|set(self.z[t][changed]))-{-1})
        pp=list(self.pose[t]);pp[i]=T;delta=0.;cols={};self.trial=(t,M)
        try:
            for j in affected:
                c=self.column(1-t,j,self.pose[1-t][j],self.surface(t,j,z,pp));cols[j]=c
                use=self.z[1-t]==j
                delta+=float(np.dot(self.weights[1-t][use],c[use]-self.C[1-t][use,j]))
        finally:self.trial=None
        return delta,cols

    def block(self,t,i):
        before=self.energy();inc=self.surface(1-t,i);best=None;criterion=-self.cfg['acceptance_tolerance'];records=[]
        for k,T in enumerate(self.pool[t][i]):
            col=self.column(t,i,T,inc);z,M,own=self.propose(t,i,col)
            back,cols=self.back(t,i,T,z,M);delta=own+back
            records.append({'k':k,'own_delta':own,'outgoing_delta':back,'joint_delta':delta})
            if delta<criterion:
                criterion=delta;best=(k,T,col,z,M,own,back,cols)
        row={'period':t+1,'asset_uid':self.H[i]['uid'],'asset_index':i,'energy_before':before,'candidates':records,'accepted':best is not None}
        if best:
            k,T,col,z,M,own,back,cols=best
            self.pose[t][i]=T.copy();self.k[t,i]=k;self.z[t]=z;self.M[t]=M;self.C[t][:,i]=col
            for j,c in cols.items():self.C[1-t][:,j]=c
            assert abs(self.energy()-before-own-back)<2e-7
            assert self.energy()<=before+2e-7
            row.update(new_k=k,own_delta=own,outgoing_delta=back)
        row['energy_after']=self.energy()
        return row
