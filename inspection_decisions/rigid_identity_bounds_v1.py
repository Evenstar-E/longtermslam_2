"""ID-blind joint rigid assignment with complete finite-domain pose partitions.

Bounds certify an explicit surrogate energy inside a declared finite workspace,
not physical correctness, sensor noise or an independent material-motion truth.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import milp, Bounds, LinearConstraint, linear_sum_assignment
from scipy.sparse import coo_matrix


def transform(points,T):return points@T[:3,:3].T+T[:3,3]

def anchor_cloud(x):
    # An actually observed historical point, NOT a changing visible centroid.
    c=x[np.argmin(np.linalg.norm(x-x.mean(0),axis=1))]
    return c,float(np.linalg.norm(x-c,axis=1).max())

def pose(r,p,c):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(r).as_matrix();T[:3,3]=p-T[:3,:3]@c
    return T


@dataclass
class Cell:
    owner:int
    lo:np.ndarray
    hi:np.ndarray
    key:int

    def center(self):return (self.lo+self.hi)/2

    def split(self,radius,start_key):
        half=(self.hi-self.lo)/2
        units=half.copy();units[:3]*=max(radius,1e-6)
        axis=int(np.argmax(units));middle=self.center()[axis]
        left=self.hi.copy();left[axis]=middle
        right=self.lo.copy();right[axis]=middle
        return [Cell(self.owner,self.lo.copy(),left,start_key),Cell(self.owner,right,self.hi.copy(),start_key+1)]


def lower_point_distances(points,history,anchor,radius,cell):
    """For EVERY SE3 in cell, returned per-point distances <= true NN distances."""
    mid=cell.center();half=(cell.hi-cell.lo)/2
    R=Rotation.from_rotvec(mid[:3]).as_matrix();p=mid[3:]
    # Inverse-query the ORIGINAL observed historical surface.
    center_dist=cKDTree(history).query((points-p)@R+anchor,workers=1)[0]
    eta=np.linalg.norm(half[3:])+2*radius*np.sin(min(np.pi,np.linalg.norm(half[:3]))/2)
    lipschitz=np.maximum(0,center_dist-eta)
    # Every possible transformed historical point lies in the sphere at p.
    # This rotation-independent bound can tighten the coarse rotation cells.
    point_to_position_box=np.linalg.norm(np.maximum(np.maximum(cell.lo[3:]-points,points-cell.hi[3:]),0),axis=1)
    sphere=np.maximum(0,point_to_position_box-radius)
    return np.maximum(lipschitz,sphere)


def rgb_hist(rgb,bins):
    return np.concatenate([np.histogram(rgb[:,i],bins=bins,range=(0,256))[0]/len(rgb) for i in range(3)])/3


class JointProblem:
    def __init__(self,history,current,shape_cost,cfg):
        self.H=history;self.C=current;self.cfg=cfg;self.n=len(current);self.m=len(history)
        self.mass=np.array([len(v['xyz']) for v in current],float);self.mass/=self.mass.sum()
        self.anchor=[];self.radius=[];self.tree=[];self.fit_calls=0;self.point_queries=0
        for h in history:
            c,r=anchor_cloud(h['xyz']);self.anchor.append(c);self.radius.append(r);self.tree.append(cKDTree(h['xyz']))
        self.anchor=np.array(self.anchor);self.radius=np.array(self.radius)
        hh=np.array([rgb_hist(h['rgb'],cfg['rgb_bins_per_channel']) for h in history]);cc=np.array([rgb_hist(c['rgb'],cfg['rgb_bins_per_channel']) for c in current])
        rgb=.5*np.abs(cc[:,None]-hh[None]).sum(-1)
        self.appearance=cfg['shape_weight']*shape_cost+cfg['rgb_weight']*rgb
        xyz=np.concatenate([v['xyz'] for v in current]);margin=max(self.radius)+cfg['distance_scale_m']
        self.position_lo=xyz.min(0)-margin;self.position_hi=xyz.max(0)+margin
        self.cells={i:Cell(i,np.r_[[-np.pi]*3,self.position_lo],np.r_[[np.pi]*3,self.position_hi],i) for i in range(self.m)}
        self.next_key=self.m;self.cell_cost={};self.trace=[]

    def in_domain(self,T,i):
        p=transform(self.anchor[i][None],T)[0]
        return bool(np.all(p>=self.position_lo-1e-7) and np.all(p<=self.position_hi+1e-7))

    def costs(self,poses):
        out=np.empty((self.n,self.m));tau=self.cfg['distance_scale_m']
        for i,T in enumerate(poses):
            R=T[:3,:3];t=T[:3,3];p=transform(self.anchor[i][None],T)[0]
            motion=self.cfg['motion_weight']*min(float(np.linalg.norm(p-self.anchor[i])),1.)
            for j,c in enumerate(self.C):
                d=self.tree[i].query((c['xyz']-t)@R,workers=1)[0];self.point_queries+=len(d)
                out[j,i]=self.cfg['geometry_weight']*np.minimum(d/tau,1).dot(np.minimum(d/tau,1))/len(d)+self.appearance[j,i]+motion
        return out

    def pair_cost(self,j,i,T):
        c=self.C[j];R=T[:3,:3];t=T[:3,3];d=self.tree[i].query((c['xyz']-t)@R,workers=1)[0];self.point_queries+=len(d)
        p=transform(self.anchor[i][None],T)[0]
        return float(self.cfg['geometry_weight']*np.mean(np.minimum(d/self.cfg['distance_scale_m'],1)**2)+self.appearance[j,i]+self.cfg['motion_weight']*min(np.linalg.norm(p-self.anchor[i]),1.))

    def fit_pose(self,i,xyz,initial):
        import open3d as o3d
        def cloud(x):
            p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(x)
            return p.voxel_down_sample(self.cfg['icp_voxel_m'])
        r=o3d.pipelines.registration.registration_icp(cloud(xyz),cloud(self.H[i]['xyz']),self.cfg['icp_distance_m'],np.linalg.inv(initial),o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.cfg['icp_iterations']))
        self.fit_calls+=1;T=np.linalg.inv(np.asarray(r.transformation))
        return T if self.in_domain(T,i) else initial.copy()

    def assign(self,cost,one_to_one=False):
        if one_to_one:
            ii,jj=linear_sum_assignment(np.c_[self.mass[:,None]*cost,np.diag(self.mass)*self.cfg['null_cost']+np.where(np.eye(self.n),0,1e6)])
            a=np.full(self.n,-1,int)
            for j,i in zip(ii,jj):
                if i<self.m:a[j]=i
            return a
        best=np.argmin(cost,axis=1);return np.where(cost[np.arange(self.n),best]<self.cfg['null_cost'],best,-1)

    def energy(self,a,poses):
        costs=self.costs(poses)
        return float(sum(self.mass[j]*(self.cfg['null_cost'] if i<0 else costs[j,i]) for j,i in enumerate(a)))

    def refine(self,a,poses,iterations):
        poses=[T.copy() for T in poses];energy=self.energy(a,poses)
        for _ in range(iterations):
            previous=energy
            for i in sorted(set(a)-{-1}):
                xyz=np.concatenate([self.C[j]['xyz'] for j in np.flatnonzero(a==i)])
                # Current->history ICP proposes ONE shared transform for all fragments.
                candidate=self.fit_pose(i,xyz,poses[i])
                if not self.in_domain(candidate,i):continue
                pp=[T.copy() for T in poses];pp[i]=candidate;e=self.energy(a,pp)
                if e<energy:poses=pp;energy=e
            aa=self.assign(self.costs(poses));ee=self.energy(aa,poses)
            if ee<=energy:a=aa;energy=ee
            if previous-energy<1e-9:break
        return {'assignment':a.tolist(),'poses':[T.tolist() for T in poses],'energy':energy}

    def bound_cost(self,cell):
        if cell.key in self.cell_cost:return self.cell_cost[cell.key]
        i=cell.owner;mid=cell.center();tau=self.cfg['distance_scale_m']
        displacement_lb=np.linalg.norm(np.maximum(np.maximum(cell.lo[3:]-self.anchor[i],self.anchor[i]-cell.hi[3:]),0))
        motion=self.cfg['motion_weight']*min(displacement_lb,1.)
        costs=[]
        for j,c in enumerate(self.C):
            d=lower_point_distances(c['xyz'],self.H[i]['xyz'],self.anchor[i],self.radius[i],cell)
            self.point_queries+=len(d)
            costs.append(self.mass[j]*(self.cfg['geometry_weight']*np.mean(np.minimum(d/tau,1)**2)+self.appearance[j,i]+motion))
        self.cell_cost[cell.key]=np.array(costs);return self.cell_cost[cell.key]

    def master(self,force=None):
        """Mixed integer lower-bound master; assignment y continuous, cell z binary.

        One chosen pose CELL per archive object. All its fragments share it.
        Within-cell pose relaxation is conservative and shrinks by subdivision.
        """
        cells=list(self.cells.values());K=len(cells);n=self.n
        costs=np.array([self.bound_cost(cell) for cell in cells]).T
        # variables y[j,k], null[j], z[k].
        zbase=n*K+n;total=zbase+K;c=np.r_[costs.ravel(),self.mass*self.cfg['null_cost'],np.zeros(K)]
        rr=[];cc=[];vv=[];lo=[];hi=[]
        def constraint(items,l,u):
            index=len(lo)
            for col,value in items:rr.append(index);cc.append(col);vv.append(value)
            lo.append(l);hi.append(u)
        for j in range(n):constraint([(j*K+k,1.) for k in range(K)]+[(n*K+j,1.)],1.,1.)
        for i in range(self.m):constraint([(zbase+k,1.) for k,cell in enumerate(cells) if cell.owner==i],0.,1.)
        for j in range(n):
            for k in range(K):constraint([(j*K+k,1.),(zbase+k,-1.)],-np.inf,0.)
        if force is not None:
            j,identity=force
            constraint([(n*K+j,1.)] if identity<0 else [(j*K+k,1.) for k,cell in enumerate(cells) if cell.owner==identity],1.,1.)
        matrix=coo_matrix((vv,(rr,cc)),shape=(len(lo),total)).tocsc()
        began=time.perf_counter();r=milp(c,integrality=np.r_[np.zeros(zbase),np.ones(K)],bounds=Bounds(np.zeros(total),np.ones(total)),constraints=LinearConstraint(matrix,np.array(lo),np.array(hi)),options={'time_limit':self.cfg['master_time_limit_s'],'mip_rel_gap':0.0})
        dual=getattr(r,'mip_dual_bound',None)
        if dual is None or not np.isfinite(dual):dual=0.0 # all terms nonnegative; timeout is not infeasibility.
        assignment=None;selected={}
        if r.x is not None:
            y=r.x[:n*K].reshape(n,K);assignment=np.full(n,-1,int)
            for j in range(n):
                k=int(np.argmax(y[j]))
                if y[j,k]>.5:assignment[j]=cells[k].owner;selected[cells[k].owner]=cells[k]
        return {'lower_bound':float(max(0,dual)),'incumbent_relaxed_cost':None if r.fun is None else float(r.fun),'assignment':assignment,'selected':selected,'status':int(r.status),'elapsed_s':time.perf_counter()-began,'cells':K}

    def solve_bounds(self,initial,monitor=None):
        best=initial;trace=[];last_master=None
        for step in range(self.cfg['max_splits_per_class']+1):
            master=self.master();last_master=master
            if master['assignment'] is not None:
                poses=[np.array(T) for T in best['poses']]
                for i,cell in master['selected'].items():
                    mid=cell.center();poses[i]=pose(mid[:3],mid[3:],self.anchor[i])
                trial=self.refine(master['assignment'],poses,1)
                if trial['energy']<best['energy']:best=trial
            record={'step':step,'lower_bound':master['lower_bound'],'upper_bound':best['energy'],'cells':len(self.cells),'master_status':master['status'],'master_elapsed_s':master['elapsed_s'],'fit_calls':self.fit_calls,'point_queries':self.point_queries}
            if master['lower_bound']>best['energy']+1e-6:raise AssertionError('Invalid lower bound exceeds feasible upper bound')
            trace.append(record)
            if monitor:monitor(record)
            if step==self.cfg['max_splits_per_class'] or best['energy']-master['lower_bound']<1e-5:break
            # Prefer selected coarse regions; no learned proposal removes other regions.
            pool=list(master['selected'].values()) or list(self.cells.values())
            def priority(cell):
                half=(cell.hi-cell.lo)/2
                return np.linalg.norm(half[3:])+self.radius[cell.owner]*min(np.linalg.norm(half[:3]),np.pi)
            target=max(pool,key=priority);children=target.split(self.radius[target.owner],self.next_key);self.next_key+=2
            del self.cells[target.key]
            for cell in children:self.cells[cell.key]=cell
        # Forced-identity lower bounds keep all continuous regions. Identity sets
        # refer to near-optimal ENERGY explanations, not physical correctness.
        threshold=best['energy']+self.cfg['near_optimal_energy_slack'];sets=[]
        for j in range(self.n):
            bounds=[];possible=[]
            for identity in [-1]+list(range(self.m)):
                r=self.master((j,identity));lb=r['lower_bound'];bounds.append({'identity':identity,'lower_bound':lb,'status':r['status']})
                if lb<=threshold+1e-7:possible.append(identity)
            sets.append({'current':j,'possible_identities':possible,'forced_lower_bounds':bounds,'energy_threshold':threshold})
        return {'best':best,'trace':trace,'identity_sets':sets,'root_position_lo':self.position_lo.tolist(),'root_position_hi':self.position_hi.tolist(),
            'leaf_cells':[{'owner':v.owner,'key':v.key,'lo':v.lo.tolist(),'hi':v.hi.tolist()} for v in self.cells.values()],
            'last_lower_bound':last_master['lower_bound'],'fit_calls':self.fit_calls,'point_queries':self.point_queries,
            'claim':'Complete continuous pose-domain partition and conservative surrogate-energy bounds. No independent physical-ID/motion certificate; common frame G not estimated.'}
