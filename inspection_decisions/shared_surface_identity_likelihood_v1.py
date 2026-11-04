"""Normalized joint mixture and conditional Gaussian-mean variational baseline.

Standard EM/VB tools; not an original theorem or complete pose/identity posterior.
Each weighted observation has one mixture factor, not two conditional map scores.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from inspection_decisions.reversible_surface_v2 import transform,inverse


def sample(x,n):
    if len(x)==0:raise ValueError('empty template')
    out=[];dist=np.full(len(x),np.inf);j=0
    for _ in range(n):
        out.append(j);dist=np.minimum(dist,np.sum((x-x[j])**2,axis=1));j=int(np.argmax(dist))
    return np.array(out)


class Model:
    def __init__(self,H,O,pools,state,cfg,method):
        self.H=H;self.O=O;self.pool=pools;self.cfg=cfg;self.method=method
        self.pose=state['poses'].copy();self.k=state['k'].copy();self.m=len(H);self.kh=cfg['history_components'];self.ku=cfg['new_components']
        self.sigma=np.array([cfg['position_sigma_m']]*3+[cfg['color_sigma']]*3)
        self.priorvar=np.array([cfg['prior_position_sigma_m']**2]*3+[cfg['prior_color_sigma']**2]*3)
        self.mean=[];self.var=[];self.prior=[];self.trace=[];self.queries=0
        for i,h in enumerate(H):
            ih=sample(h['xyz'],self.kh);hist=np.c_[h['xyz'][ih],h['rgb'][ih]]
            xx=[]
            for t in [0,1]:
                idx=np.flatnonzero(state[f'labels{t+1}']==i)
                if len(idx):xx.append(np.c_[inverse(O[t]['xyz'][idx],self.pose[t,i]),O[t]['rgb'][idx]])
            data=np.concatenate(xx) if xx else hist
            # All observations enter likelihood; FPS is only the fixed-size representation.
            new=data[sample(data[:,:3],self.ku)]
            prior=np.r_[h['xyz'].mean(0),h['rgb'].mean(0)]
            n=max(1.,sum(O[t]['mass'][state[f'labels{t+1}']==i].sum() for t in [0,1])/self.ku)
            v=1/(1/self.priorvar+n/self.sigma**2) if method=='shared_vb' else np.zeros(6)
            self.mean.append(np.r_[hist,new]);self.var.append(np.tile(v,(self.ku,1)));self.prior.append(prior)
        self.mean=np.array(self.mean);self.var=np.array(self.var);self.prior=np.array(self.prior)
        self.categories=1+max(max(int(o['class'].max()) for o in O),max(h['class'] for h in H))
        allx=np.concatenate([o['xyz'] for o in O]);self.low=allx.min(0)-1;self.high=allx.max(0)+1
        self.log_out=np.log(cfg['outlier_mass'])-np.log(np.prod(self.high-self.low))-np.log(self.categories)
        self.component_logweight=np.r_[np.full(self.kh,np.log((1-cfg['new_mass'])/self.kh)),np.full(self.ku,np.log(cfg['new_mass']/self.ku))]
        self.log_asset=np.log((1-cfg['outlier_mass'])/self.m)
        self.L=[self.columns(t) for t in [0,1]]

    def components(self,t,i,T=None,obs=None):
        o=self.O[t] if obs is None else obs;T=self.pose[t,i] if T is None else T
        center=self.mean[i];world=np.c_[transform(center[:,:3],T),center[:,3:]]
        f=np.c_[o['xyz'],o['rgb']]/self.sigma;g=world/self.sigma
        d=cdist(f,g,'sqeuclidean');self.queries+=len(f)*len(g)
        lp=-.5*d-3*np.log(2*np.pi)-np.log(self.sigma).sum()+self.component_logweight+self.log_asset
        if self.method=='shared_vb':lp[:,self.kh:]-=.5*np.sum(self.var[i]/self.sigma**2,axis=1)
        correct=self.cfg['semantic_correct'];sem=np.where(o['class']==self.H[i]['class'],correct,(1-correct)/max(1,self.categories-1))
        return lp+np.log(sem)[:,None]

    def columns(self,t,obs=None):return np.column_stack([logsumexp(self.components(t,i,obs=obs),axis=1) for i in range(self.m)])

    def partition(self,L):return logsumexp(np.c_[L,np.full(len(L),self.log_out)],axis=1)

    def kl(self):
        if self.method!='shared_vb':return 0.
        d=self.mean[:,self.kh:]-self.prior[:,None,:];v=self.var
        return float(.5*np.sum((v+d*d)/self.priorvar-1+np.log(self.priorvar/v)))

    def objective(self):return float(sum(np.dot(o['mass'],self.partition(L)) for o,L in zip(self.O,self.L))-self.kl())

    def update_surface(self):
        if self.method=='frozen_shared':return
        before=self.objective();new=self.mean.copy();v=self.var.copy();counts=np.zeros((2,self.m,self.ku))
        for i in range(self.m):
            mass=np.zeros(self.ku);sums=np.zeros((self.ku,6))
            for t in [0,1]:
                lp=self.components(t,i)[:,self.kh:];r=np.exp(lp-self.partition(self.L[t])[:,None])*self.O[t]['mass'][:,None]
                y=np.c_[inverse(self.O[t]['xyz'],self.pose[t,i]),self.O[t]['rgb']]
                n=r.sum(0);counts[t,i]=n;mass+=n;sums+=r.T@y
            if self.method=='shared_vb':
                v[i]=1/(1/self.priorvar+mass[:,None]/self.sigma**2)
                new[i,self.kh:]=v[i]*(self.prior[i]/self.priorvar+sums/self.sigma**2)
            else:
                ok=mass>1e-12;new[i,self.kh:][ok]=sums[ok]/mass[ok,None]
        self.mean=new;self.var=v;self.L=[self.columns(t) for t in [0,1]];after=self.objective()
        assert after>=before-1e-5,(before,after)
        self.trace.append({'kind':'surface','before':before,'after':after});self.counts=counts

    def pose_sweep(self,monitor=None):
        for t in [0,1]:
            for i in range(self.m):
                before=self.objective();others=self.L[t].copy();others[:,i]=-np.inf
                other=self.partition(others);best=np.dot(self.O[t]['mass'],np.logaddexp(other,self.L[t][:,i]));chosen=None;records=[]
                for k,T in enumerate(self.pool[t][i]):
                    col=logsumexp(self.components(t,i,T),axis=1)
                    value=float(np.dot(self.O[t]['mass'],np.logaddexp(other,col)));records.append(value)
                    if value>best+1e-7:best=value;chosen=(k,T,col)
                if chosen:
                    k,T,col=chosen;self.k[t,i]=k;self.pose[t,i]=T;self.L[t][:,i]=col
                after=self.objective();assert after>=before-1e-5
                row={'kind':'pose','period':t+1,'asset_uid':self.H[i]['uid'],'asset_index':i,'before':before,'after':after,'accepted':chosen is not None,'candidate_values':records,'selected_k':int(self.k[t,i])}
                self.trace.append(row)
                if monitor:monitor(row)

    def readout(self,t,obs=None):
        o=self.O[t] if obs is None else obs;L=self.L[t] if obs is None else self.columns(t,obs)
        a=np.argmax(L,axis=1);score=L[np.arange(len(L)),a];a[score<=self.log_out]=-1
        confidence=np.exp(np.maximum(score,self.log_out)-self.partition(L))
        support=np.zeros(len(a),bool)
        for i in np.unique(a[a>=0]):
            idx=np.flatnonzero(a==i);world=transform(self.mean[i,:,:3],self.pose[t,i])
            support[idx]=np.min(cdist(o['xyz'][idx],world),axis=1)<self.cfg['distance_scale_m']
        return a,confidence,support
