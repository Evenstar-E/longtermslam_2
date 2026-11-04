"""Finite-pool pose/ownership blocks with exact outgoing-surface consequences.

Candidate generation and generic block-coordinate descent are established tools.
This experiment isolates ignoring vs evaluating outgoing observation dependencies.
"""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.reversible_surface_v2 import costs, inverse, transform


def rigid_projection(T):
    T=np.array(T,float).copy();u,_,v=np.linalg.svd(T[:3,:3]);d=np.eye(3)
    d[2,2]=np.linalg.det(u@v);T[:3,:3]=u@d@v;T[3]=[0,0,0,1]
    return T


class Problem:
    def __init__(self, observations, history, pools, initial_labels, cfg, method):
        self.O=observations;self.H=history;self.pool=pools;self.cfg=cfg;self.method=method
        self.bidir=method in ['pool_detached','pool_coupled'];self.coupled=method=='pool_coupled'
        self.pose=[[p[0].copy() for p in pp] for pp in pools]
        self.k=np.zeros((2,len(history)),int);self.z=[a.copy() for a in initial_labels]
        self.initial_labels=[a.copy() for a in initial_labels]
        self.weights=[o['mass']/o['mass'].sum() for o in observations]
        self.scene_trees=[cKDTree(o['xyz']) for o in observations]
        self.null=cfg['null_cost'];self.point_queries=0;self.column_calls=0;self.trace=[]
        self.C=[]
        for t in [0,1]:
            cols=[]
            for i in range(len(history)):
                inc=self.surface(1-t,i) if t==1 or self.bidir else None
                cols.append(self.column(t,i,self.pose[t][i],inc))
            self.C.append(np.column_stack(cols))
        self.initial_energy=self.energy()

    def surface(self,t,i,labels=None,poses=None):
        labels=self.z[t] if labels is None else labels
        poses=self.pose[t] if poses is None else poses
        idx=np.flatnonzero(labels==i);q=inverse(self.O[t]['xyz'][idx],poses[i])
        return {'xyz':q,'rgb':self.O[t]['rgb'][idx],'source_voxel':idx,
                'tree':cKDTree(q) if len(q) else None}

    def column(self,t,i,T,inc):
        """Exact null-clipped column with conservative spatial acceleration.

        Every pose is tested. Outside the transformed surface-enclosing sphere
        plus tau, the nonnegative geometry term alone exceeds null; those points
        provably cannot improve the null-clipped loss. No identity/pose pruning.
        """
        o=self.O[t];h=self.H[i];x=h['xyz']
        if inc is not None and len(inc['xyz']):x=np.concatenate([x,inc['xyz']])
        center=(x.min(0)+x.max(0))/2
        radius=np.linalg.norm(x-center,axis=1).max()+self.cfg['distance_scale_m']+1e-5
        idx=np.array(sorted(self.scene_trees[t].query_ball_point(transform(center[None],T)[0],radius)),int)
        out=np.full(len(o['xyz']),self.null,np.float32);self.column_calls+=1
        if len(idx):
            sub={k:o[k][idx] for k in ['xyz','rgb','class']}
            cc,_,_,_=costs(sub,[T],[h],None if inc is None else [inc],self.cfg)
            out[idx]=np.minimum(cc[:,0],self.null);self.point_queries+=len(idx)
        return out

    def assigned(self,t,z=None,C=None):
        z=self.z[t] if z is None else z;C=self.C[t] if C is None else C
        val=np.full(len(z),self.null,float);ok=z>=0
        val[ok]=C[np.flatnonzero(ok),z[ok]]
        return val

    def energy(self):
        return float(sum(np.dot(self.weights[t],self.assigned(t)) for t in ([0,1] if self.bidir else [1])))

    def label_proposal(self,t,i,col):
        # Only the old/new influence region and currently owned points are revised.
        # Unrelated source observations are not wholesale reset by one pose update.
        region=np.flatnonzero((self.z[t]==i)|(self.C[t][:,i]<self.null)|(col<self.null))
        z=self.z[t].copy();old=self.assigned(t)
        C=self.C[t][region].copy();C[:,i]=col[region]
        a=np.argmin(C,axis=1);value=C[np.arange(len(region)),a];a[value>=self.null]=-1
        z[region]=a
        delta=float(np.dot(self.weights[t][region],value-old[region]))
        return z,delta,region

    def backward(self,t,i,T,z):
        other=1-t;changed=z!=self.z[t]
        affected=sorted(({i}|set(z[changed])|set(self.z[t][changed]))-{-1})
        pp=list(self.pose[t]);pp[i]=T;cols={};delta=0.
        for j in affected:
            inc=self.surface(t,j,z,pp)
            col=self.column(other,j,self.pose[other][j],inc);cols[j]=col
            mask=self.z[other]==j
            delta+=float(np.dot(self.weights[other][mask],col[mask]-self.C[other][mask,j]))
        return delta,cols,affected

    def block(self,t,i):
        inc=self.surface(1-t,i) if t==1 or self.bidir else None
        best=None;best_selection=0.;records=[];before=self.energy()
        candidates=range(len(self.pool[t][i])) if self.method!='selected_forward' else [0]
        for k in candidates:
            T=self.pool[t][i][k];col=self.column(t,i,T,inc)
            z,own,region=self.label_proposal(t,i,col)
            back,backcols,affected=self.backward(t,i,T,z) if self.bidir else (0.,{},[])
            joint=own+back;selection=joint if self.coupled else own
            records.append({'k':int(k),'own_delta':own,'outgoing_delta':back,'joint_delta':joint,
                            'changed_voxels':int(np.sum(z!=self.z[t])),'affected_assets':len(affected)})
            if selection<best_selection-self.cfg['acceptance_tolerance']:
                best_selection=selection;best=(k,T,col,z,own,back,backcols)
        row={'period':t+1,'asset_index':i,'asset_uid':self.H[i]['uid'],
             'candidates':records,'energy_before':before,'accepted':best is not None,
             'old_k':int(self.k[t,i])}
        if best is not None:
            k,T,col,z,own,back,backcols=best
            self.pose[t][i]=T.copy();self.k[t,i]=k;self.z[t]=z;self.C[t][:,i]=col
            for j,c in backcols.items():self.C[1-t][:,j]=c
            after=self.energy()
            assert abs(after-before-own-back)<2e-7,(after,before,own,back)
            if self.coupled:assert after<=before+2e-7
            row.update(new_k=int(k),own_delta=own,outgoing_delta=back,energy_after=after)
        else:row.update(new_k=int(self.k[t,i]),own_delta=0.,outgoing_delta=0.,energy_after=before)
        return row

    def solve(self,monitor=None):
        for sweep in range(self.cfg['sweeps']):
            start=self.energy();accepted=0
            for t in ([0,1] if self.bidir else [1]):
                for i in range(len(self.H)):
                    row=self.block(t,i);row['sweep']=sweep;self.trace.append(row);accepted+=row['accepted']
                    if monitor:monitor(row,self)
            if monitor:monitor({'sweep_complete':sweep,'energy_before':start,'energy_after':self.energy(),
                                'accepted_blocks':int(accepted)},self)
        return self

    def verify_cache(self):
        """Rebuild from source labels and poses to check every dependency update."""
        maximum=0.
        for t in [0,1]:
            for i in range(len(self.H)):
                inc=self.surface(1-t,i) if t==1 or self.bidir else None
                c=self.column(t,i,self.pose[t][i],inc)
                maximum=max(maximum,float(np.max(np.abs(c-self.C[t][:,i]))))
        assert maximum<1e-6,maximum
        return maximum
