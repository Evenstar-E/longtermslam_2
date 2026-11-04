"""Cross-fragment rigidity cuts for the SAME observed-surface surrogate energy.

This is a bound-strengthening ablation, not another score/threshold setting.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from inspection_decisions.rigid_identity_bounds_v1 import JointProblem


class RigidityCutsProblem(JointProblem):
    def build_conflicts(self):
        self.conflicts=[];tau=self.cfg['distance_scale_m'];trees=[cKDTree(c['xyz']) for c in self.C]
        # Bounding-box diagonal and twice anchor radius are BOTH upper bounds
        # on archive diameter, so their minimum remains an upper bound.
        diameter=np.array([min(np.linalg.norm(np.ptp(h['xyz'],axis=0)),2*self.radius[i]) for i,h in enumerate(self.H)])
        for j in range(self.n):
            for k in range(j+1,self.n):
                separation=float(trees[k].query(self.C[j]['xyz'],workers=1)[0].min())
                for i in range(self.m):
                    if separation>diameter[i]+2*tau+1e-9:
                        self.conflicts.append((i,j,k,separation,float(diameter[i])))
        return self.conflicts

    def master(self,force=None):
        if not hasattr(self,'conflicts'):self.build_conflicts()
        cells=list(self.cells.values());K=len(cells);n=self.n;m=self.m
        total_cost=np.array([self.bound_cost(cell) for cell in cells]).T
        fixed=np.empty_like(total_cost)
        for k,cell in enumerate(cells):
            i=cell.owner;distance=np.linalg.norm(np.maximum(np.maximum(cell.lo[3:]-self.anchor[i],self.anchor[i]-cell.hi[3:]),0))
            fixed[:,k]=self.mass*(self.appearance[:,i]+self.cfg['motion_weight']*min(distance,1.))
        geo=np.maximum(0,total_cost-fixed)
        zbase=n*K+n;gbase=zbase+K;total=gbase+n*m
        objective=np.r_[fixed.ravel(),self.mass*self.cfg['null_cost'],np.zeros(K),np.ones(n*m)]
        rows=[];cols=[];values=[];lower=[];upper=[]
        def constraint(items,lo,hi):
            row=len(lower)
            for col,value in items:rows.append(row);cols.append(col);values.append(value)
            lower.append(lo);upper.append(hi)
        owner_cells={i:[k for k,c in enumerate(cells) if c.owner==i] for i in range(m)}
        for j in range(n):constraint([(j*K+k,1.) for k in range(K)]+[(n*K+j,1.)],1.,1.)
        for i,ks in owner_cells.items():constraint([(zbase+k,1.) for k in ks],0.,1.)
        for j in range(n):
            for k in range(K):constraint([(j*K+k,1.),(zbase+k,-1.)],-np.inf,0.)
            for i,ks in owner_cells.items():
                g=gbase+j*m+i
                constraint([(g,1.)]+[(j*K+k,-geo[j,k]) for k in ks],0.,np.inf)
                constraint([(g,1.)]+[(j*K+k,-self.cfg['geometry_weight']*self.mass[j]) for k in ks],-np.inf,0.)
        if force is not None:
            j,i=force;constraint([(n*K+j,1.)] if i<0 else [(j*K+k,1.) for k in owner_cells[i]],1.,1.)
        for i,j,l,_,_ in self.conflicts:
            # If both fragments use i, at least one has ALL points beyond tau.
            # Normalized geometry losses must sum to >=1. Point distances are
            # computed on entire fragments, not their centroids or sampled tips.
            items=[(gbase+j*m+i,1/(self.cfg['geometry_weight']*self.mass[j])),(gbase+l*m+i,1/(self.cfg['geometry_weight']*self.mass[l]))]
            items += [(j*K+k,-1.) for k in owner_cells[i]]+[(l*K+k,-1.) for k in owner_cells[i]]
            constraint(items,-1.,np.inf)
        A=coo_matrix((values,(rows,cols)),shape=(len(lower),total)).tocsc();start=time.perf_counter()
        r=milp(objective,integrality=np.r_[np.ones(n*K),np.zeros(n),np.ones(K),np.zeros(n*m)],bounds=Bounds(np.zeros(total),np.ones(total)),constraints=LinearConstraint(A,np.array(lower),np.array(upper)),options={'time_limit':self.cfg['master_time_limit_s'],'mip_rel_gap':0.0})
        dual=getattr(r,'mip_dual_bound',None)
        if dual is None or not np.isfinite(dual):dual=0.
        return {'lower_bound':float(max(0,dual)),'incumbent_relaxed_cost':None if r.fun is None else float(r.fun),'status':int(r.status),'elapsed_s':time.perf_counter()-start,'cells':K,'rigidity_cuts':len(self.conflicts)}
