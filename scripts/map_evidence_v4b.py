"""Bounded censored-map feasibility with shared frame calibration parameters.

MILP and set-membership reasoning are established tools; originality unclaimed.
Labels: 0=no substantial axis motion supported, 1=change, -1=unresolved.
"""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix


class Model:
    def __init__(self,inp,cfg):
        self.cfg=cfg;n=inp['assets'];self.n=n;self.q=np.array(inp['assignment']);frames=inp['frames']
        self.d0=6*n;self.u0=9*n;self.nu=7*frames;self.z0=self.u0+self.nu;self.ad0=self.z0+n;self.au0=self.ad0+3*n;self.size=self.au0+self.nu
        self.lower=np.full(self.size,-20.);self.upper=np.full(self.size,20.);self.lower[self.d0:self.u0]=-cfg['motion_bound_m'];self.upper[self.d0:self.u0]=cfg['motion_bound_m'];self.lower[self.u0:self.z0]=-1;self.upper[self.u0:self.z0]=1;self.lower[self.z0:]=0;self.upper[self.z0:self.ad0]=1;self.upper[self.ad0:]=20
        self.integrality=np.zeros(self.size);self.integrality[self.z0:self.ad0]=1
        rows=[];low=[];high=[]
        def add(items,l=-np.inf,h=np.inf):
            row=np.zeros(self.size)
            for k,v in items:row[k]+=v
            rows.append(row);low.append(l);high.append(h)
        for j in range(n):
            for a in range(3):
                add([(6*j+2*a+1,1),(6*j+2*a,-1)],*cfg['box_width_prior_m'])
                d=self.d0+3*j+a;z=self.z0+j;extra=cfg['motion_bound_m']-cfg['motion_axis_threshold_m']
                add([(d,1),(z,-extra)],h=cfg['motion_axis_threshold_m']);add([(d,-1),(z,-extra)],h=cfg['motion_axis_threshold_m'])
                add([(d,1),(self.ad0+3*j+a,-1)],h=0);add([(d,-1),(self.ad0+3*j+a,-1)],h=0)
            if j not in self.q:self.lower[self.d0+3*j:self.d0+3*j+3]=0;self.upper[self.d0+3*j:self.d0+3*j+3]=0;self.upper[self.z0+j]=0
        add([(self.z0+j,1) for j in range(n)],h=cfg['max_changed_assets'])
        for k in range(self.nu):
            add([(self.u0+k,1),(self.au0+k,-1)],h=0);add([(self.u0+k,-1),(self.au0+k,-1)],h=0)
        for current,records in ((False,inp['history']),(True,inp['current'])):
            for record in records:
                j=record['asset'];frame=record['frame']
                for e in record['endpoints']:
                    a=e['axis'];side=e['side'];items=[(6*j+2*a+side,1)]
                    if current:items.append((self.d0+3*j+a,1))
                    items.extend((self.u0+7*frame+k,v) for k,v in enumerate(e['jacobian']))
                    tol=cfg['measurement_tolerance_m']+e.get('tolerance_extra_m',0.);v=e['value']
                    add(items,l=v-tol if (e['tight'] or side) else -np.inf,h=v+tol if (e['tight'] or not side) else np.inf)
        self.constraint=LinearConstraint(csr_matrix(np.array(rows)),np.array(low),np.array(high))

    def solve(self,objective=None,bounds=None):
        if objective is None:
            objective=np.zeros(self.size);objective[self.z0:self.ad0]=100;objective[self.ad0:self.au0]=.01;objective[self.au0:]=.001
        low,high=(self.lower,self.upper) if bounds is None else bounds
        return milp(objective,integrality=self.integrality,bounds=Bounds(low,high),constraints=self.constraint,options=dict(time_limit=self.cfg['milp_time_limit_s'],mip_rel_gap=1e-8))


def infer(inp,cfg,seed):
    begin=time.perf_counter();model=Model(inp,cfg);nq=len(model.q);tau=cfg['motion_axis_threshold_m'];base=model.solve();calls=1
    labels={m:np.full(nq,-1,int) for m in cfg['methods']};states=dict(base_status=int(base.status),timeouts=0,sampling_accepted=0)
    if base.status!=0:return dict(labels={k:v.tolist() for k,v in labels.items()},seconds=time.perf_counter()-begin,calls=calls,**states)
    d=base.x[model.d0:model.u0].reshape(model.n,3);labels['single_solution']=(np.max(np.abs(d[model.q]),axis=1)>tau+1e-5).astype(int)
    intervals=np.full((nq,3,2),np.nan)
    for i,j in enumerate(model.q):
        exact=True
        for a in range(3):
            for side in (0,1):
                obj=np.zeros(model.size);obj[model.d0+3*j+a]=1 if side==0 else -1;r=model.solve(obj);calls+=1
                if r.status!=0:exact=False;states['timeouts']+=int(r.status==1)
                else:intervals[i,a,side]=r.x[model.d0+3*j+a]
        if not exact:continue
        low=intervals[i,:,0];high=intervals[i,:,1];possible_change=bool(np.any((low<-tau-1e-5)|(high>tau+1e-5)))
        box_certain_change=bool(np.any((low>tau+1e-5)|(high<-tau-1e-5)))
        labels['coordinate_intervals'][i]=1 if box_certain_change else (0 if not possible_change else -1)
        lb=model.lower.copy();ub=model.upper.copy();lb[model.d0+3*j:model.d0+3*j+3]=-tau;ub[model.d0+3*j:model.d0+3*j+3]=tau;r=model.solve(np.zeros(model.size),(lb,ub));calls+=1
        if r.status==2:labels['joint_feasibility'][i]=1
        elif r.status==0:labels['joint_feasibility'][i]=-1 if possible_change else 0
        else:states['timeouts']+=int(r.status==1)
    # Ordinary bounded calibration perturbation around a fitted feasible point.
    # This is a heuristic sensitivity sample, not calibrated posterior sampling.
    rng=np.random.default_rng(seed+444);sampled=[]
    for _ in range(cfg['sampling_objectives']):
        lb=model.lower.copy();ub=model.upper.copy();draw=np.clip(base.x[model.u0:model.z0]+rng.uniform(-.35,.35,model.nu),-1,1);lb[model.u0:model.z0]=draw;ub[model.u0:model.z0]=draw
        obj=np.zeros(model.size);obj[model.d0:model.u0]=rng.normal(size=3*model.n);r=model.solve(obj,(lb,ub));calls+=1
        if r.status==0:sampled.append(np.max(np.abs(r.x[model.d0:model.u0].reshape(model.n,3)[model.q]),axis=1)>tau+1e-5)
        elif r.status==1:states['timeouts']+=1
    states['sampling_accepted']=len(sampled)
    if sampled:
        sampled=np.array(sampled);labels['sampled_nuisance']=np.where(sampled.all(0),1,np.where((~sampled).all(0),0,-1))
    return dict(labels={k:v.tolist() for k,v in labels.items()},intervals=intervals.tolist(),representative_delta=d[model.q].tolist(),seconds=time.perf_counter()-begin,calls=calls,**states)
