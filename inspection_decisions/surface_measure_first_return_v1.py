"""Continuous thin surfaces + local unobserved optical support, no GT inputs.

Finite empirical-Bayes dictionary; conditional variational score, NOT exact CRM
evidence. Current cause responsibilities are proposed from the H posterior and
recomputed with geometry. Gradients include their derivatives. Unknown support
is piecewise-constant optical voxels; its boundary events are not smooth.
"""
import numpy as np
import torch
from scipy.special import ndtr, gammaln, digamma, logsumexp
from scipy.spatial.transform import Rotation
from inspection_decisions.first_return_gamma_material_v1 import ray_stats
from inspection_decisions.sparse_material_evidence_v1 import rotation

DT = torch.float64


def ray_data(frames, cfg):
    """All foreground rays exact; only remaining background is quadrature."""
    O, D, S, W, PO, PD, PS, labels = [], [], [], [], [], [], [], []
    h, w, stride = cfg['height'], cfg['width'], cfg['exposure_stride']
    assert h % stride == w % stride == 0
    blocks=np.arange(h*w).reshape(h//stride,stride,w//stride,stride).transpose(0,2,1,3).reshape(-1,stride**2)
    yy,xx=np.mgrid[:stride,:stride];order=np.argsort(((yy-(stride-1)/2)**2+(xx-(stride-1)/2)**2).ravel(),kind='stable')
    blocks=blocks[:,order]
    for f in frames:
        valid = np.isfinite(f['distance']) & (f['distance'] > 0)
        foreground=np.flatnonzero(valid&(f['owner']>=0))
        background=(valid&(f['owner']<0))[blocks];count=background.sum(1);active=count>0
        sampled=blocks[active,background[active].argmax(1)]
        ii=np.r_[foreground,sampled];weights=np.r_[np.ones(len(foreground)),count[active]]
        O.append(np.broadcast_to(f['origin'], (len(ii), 3)))
        D.append(f['direction'][ii]); S.append(f['distance'][ii]); W.append(weights)
        assert weights.sum()==valid.sum() and len(np.unique(ii))==len(ii)
        ii = np.flatnonzero(valid & (f['owner'] >= 0))
        PO.append(np.broadcast_to(f['origin'], (len(ii), 3)))
        PD.append(f['direction'][ii]); PS.append(f['distance'][ii]); labels.append(f['owner'][ii])
    return dict(O=np.concatenate(O), D=np.concatenate(D), s=np.concatenate(S), weight=np.concatenate(W),
                PO=np.concatenate(PO), PD=np.concatenate(PD), ps=np.concatenate(PS), owner=np.concatenate(labels),
                raw_pixels=sum(f['pixel_count'] for f in frames), quadrature_rays=sum(map(len, S)))


def local_data(data, T, slot=None):
    R, t = T[:3,:3], T[:3,3]
    ix = data['owner'] == slot
    return dict(O=(data['O']-t)@R, D=data['D']@R, s=data['s'], weight=data['weight'],
                P=(data['PO'][ix]+data['PD'][ix]*data['ps'][ix,None]-t)@R,
                U=data['PD'][ix]@R)


def build_dictionary(frames, T, slot, cfg):
    """Measured local normals gate offset modes; no density-as-face inference."""
    groups = [[] for _ in range(3)]
    height, width = cfg['height'], cfg['width']
    for f in frames:
        points = (f['origin']+f['direction']*f['distance'][:,None]-T[:3,3])@T[:3,:3]
        p = points.reshape(height,width,3); lab=f['owner'].reshape(height,width)
        dx=p[1:-1,2:]-p[1:-1,:-2];dy=p[2:,1:-1]-p[:-2,1:-1]
        n=np.cross(dx,dy); norm=np.linalg.norm(n,axis=2)
        n=n/np.maximum(norm[:,:,None],1e-14); axis=np.abs(n).argmax(2)
        ok=(lab[1:-1,1:-1]==slot)&(lab[1:-1,2:]==slot)&(lab[1:-1,:-2]==slot)&(lab[2:,1:-1]==slot)&(lab[:-2,1:-1]==slot)
        depth=(f['distance']/np.exp(f['depth_jacobian_log'])).reshape(height,width)
        # Reject depth discontinuities, not ordinary tangential pixel spacing.
        ok &= (norm>1e-12)&(abs(depth[1:-1,2:]-depth[1:-1,:-2])<cfg['surface_depth_edge_m'])&(abs(depth[2:,1:-1]-depth[:-2,1:-1])<cfg['surface_depth_edge_m'])
        ok &= np.max(abs(n),axis=2)>=cfg['plane_axis_cosine_min']
        for a in range(3): groups[a].append(p[1:-1,1:-1][ok&(axis==a)])
    centers=[];normals=[];footprints=[];counts=[]
    for a, values in enumerate(groups):
        points=np.concatenate(values) if values else np.empty((0,3))
        if not len(points): continue
        points=points[np.argsort(points[:,a])]
        split=np.flatnonzero(np.diff(points[:,a])>cfg['surface_mode_gap_m'])+1
        for patch in np.split(points,split):
            if len(patch)<cfg['plane_min_points']: continue
            center=patch.mean(0);_,_,V=np.linalg.svd(patch-center,full_matrices=False);normal=V[-1]
            if normal[a]<0:normal=-normal
            if abs(normal[a])<cfg['plane_axis_cosine_min']:continue
            tangent=(patch-center)-((patch-center)@normal)[:,None]*normal
            footprint=max(float(np.linalg.norm(tangent,axis=1).max()),cfg['surface_footprint_floor_m'])
            centers.append(center);normals.append(normal);footprints.append(footprint);counts.append(len(patch))
    C=np.array(centers).reshape(-1,3);N=np.array(normals).reshape(-1,3);ell=np.array(footprints)
    precision=np.array([(np.eye(3)-np.outer(n,n))/l**2+np.outer(n,n)/cfg['surface_epsilon_m']**2 for n,l in zip(N,ell)]).reshape(-1,3,3)
    # Fixed total surface prior mass; adding proposals does not add mass.
    area=ell**2;mass=cfg['surface_prior_mass']*area/area.sum() if len(area) else area
    q=int(round(2*cfg['hazard_extent_m']/cfg['hazard_voxel_m']))
    return dict(center=C, normal=N, precision=precision, footprint=ell, surface_mass=mass,
                V=q**3,q=q, source_points=counts, empirical_Bayes_H_dictionary=True)


def cdf_difference(lo,hi):
    return np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))


def surface_terms(local, model, cfg):
    C,N,Q=model['center'],model['normal'],model['precision'];F=len(C)
    if not F:return np.empty((len(local['P']),0)),np.empty(0)
    v=local['P'][:,None,:]-C;u=local['U']
    incidence=np.sqrt((u@N.T)**2+cfg['incidence_floor']**2)
    logK=np.log(incidence)-np.log(np.sqrt(2*np.pi)*cfg['surface_epsilon_m'])-.5*np.einsum('nfi,fij,nfj->nf',v,Q,v,optimize=True)
    O=local['O'];D=local['D'];v=O[:,None,:]-C
    aa=np.einsum('ni,fij,nj->nf',D,Q,D,optimize=True)
    bb=np.einsum('ni,fij,nfj->nf',D,Q,v,optimize=True)
    cc=np.einsum('nfi,fij,nfj->nf',v,Q,v,optimize=True)
    lo=bb/np.sqrt(aa);hi=lo+np.sqrt(aa)*local['s'][:,None]
    incidence=np.sqrt((D@N.T)**2+cfg['incidence_floor']**2)
    exposure=incidence/(cfg['surface_epsilon_m']*np.sqrt(aa))*np.exp(-.5*np.maximum(cc-bb*bb/aa,0))*cdf_difference(lo,hi)
    return logK,np.einsum('nf,n->f',exposure,local['weight'])


def voxel_ids(P,model,cfg):
    xyz=np.floor((P+cfg['hazard_extent_m'])/cfg['hazard_voxel_m']).astype(int)
    if len(xyz) and not ((xyz>=0)&(xyz<model['q'])).all():raise ValueError('Own hit outside certified domain')
    return np.ravel_multi_index(xyz.T,(model['q'],)*3)


def unknown_exposure(data,T,cfg):
    R,t=T[:3,:3],T[:3,3];O=(data['O']-t)@R;D=data['D']@R
    # Each image has identical block weights; use separately for generality.
    result=np.zeros(int(round(2*cfg['hazard_extent_m']/cfg['hazard_voxel_m']))**3)
    for weight in np.unique(data['weight']):
        ix=data['weight']==weight
        L,_=ray_stats(O[ix],D[ix],data['s'][ix],np.zeros(ix.sum(),bool),cfg['hazard_extent_m'],cfg['hazard_voxel_m'])
        result+=weight*L[1]/cfg['hazard_voxel_m']
    return result


def causes(logK,ids,a,b,model,cfg):
    V=model['V'];e=digamma(a)-np.log(b)
    logits=np.c_[logK+e[V:][None],e[ids]-np.log(cfg['hazard_voxel_m'])]
    norm=logsumexp(logits,axis=1);R=np.exp(logits-norm[:,None]);n=np.r_[np.bincount(ids,weights=R[:,-1],minlength=V),R[:,:-1].sum(0)]
    # Equivalent to sum R(logK-logR), avoiding log(0) for vanishing kernels.
    entropy_kernel=float(norm.sum()-R[:,:-1].sum(0)@e[V:]-R[:,-1]@e[ids])
    assert abs(n.sum()-len(ids))<1e-7
    return n,entropy_kernel


def history_posterior(data,T,slot,model,cfg):
    local=local_data(data,T,slot);K,A=surface_terms(local,model,cfg);ids=voxel_ids(local['P'],model,cfg)
    L=np.r_[unknown_exposure(data,T,cfg),A]
    reserve=cfg['surface_prior_mass']/model['V'] if not len(model['center']) else 0.
    mass=np.r_[np.full(model['V'],cfg['gamma_shape']+reserve),model['surface_mass']]
    # Optical thickness theta = h * old hazard; rate=old_rate/h=1.
    rate=np.full(len(mass),cfg['gamma_rate_m']/cfg['hazard_voxel_m'])
    a=mass.copy();b=rate+L;trace=[]
    for _ in range(cfg['history_vb_iterations']):
        n,c=causes(K,ids,a,b,model,cfg);a=mass+n
        value=c+np.sum(gammaln(a)-gammaln(mass)+mass*np.log(rate)-a*np.log(b))
        trace.append(float(value))
    return a,b,dict(ELBO_trace=trace,history_hits=len(ids),features=len(mass),surface_kernels=len(model['center']),history_exposure=float(L.sum()))


def candidate_stats(data,T,slot,model,a,b,cfg,unknown_A=None):
    local=local_data(data,T,slot);K,As=surface_terms(local,model,cfg);ids=voxel_ids(local['P'],model,cfg)
    A=np.r_[unknown_exposure(data,T,cfg) if unknown_A is None else unknown_A,As]
    n,c=causes(K,ids,a,b,model,cfg)
    mean=a/b;logmean=np.log(mean)
    pos=-float(logsumexp(np.c_[K+logmean[model['V']:][None],logmean[ids]-np.log(cfg['hazard_voxel_m'])],axis=1).sum())
    return dict(n=n,A=A,c=c,mean=float(A@mean+pos),positive=pos,hit_count=len(ids))


def gamma_integral(a,b,n,A):
    return gammaln(a+n)-gammaln(a)+a*np.log(b)-(a+n)*np.log(b+A)


def cost(a,b,S1,S2,method):
    if method=='surface_shared':return float(-gamma_integral(a,b,S1['n']+S2['n'],S1['A']+S2['A']).sum()-S1['c']-S2['c'])
    if method=='surface_independent':return float(-gamma_integral(a,b,S1['n'],S1['A']).sum()-gamma_integral(a,b,S2['n'],S2['A']).sum()-S1['c']-S2['c'])
    return float(S1['mean' if method=='surface_mean' else 'positive']+S2['mean' if method=='surface_mean' else 'positive'])


def pair_costs(a,b,S1,S2,methods):
    n1=np.array([s['n'] for s in S1]);n2=np.array([s['n'] for s in S2]);A1=np.array([s['A'] for s in S1]);A2=np.array([s['A'] for s in S2])
    u=-gamma_integral(a,b,n1,A1).sum(1)-np.array([s['c'] for s in S1]);v=-gamma_integral(a,b,n2,A2).sum(1)-np.array([s['c'] for s in S2])
    independent=u[:,None]+v[None,:];shared=independent.copy()
    for k in range(len(a)):
        ia=np.flatnonzero((n1[:,k]+A1[:,k])>0);ib=np.flatnonzero((n2[:,k]+A2[:,k])>0)
        if not len(ia) or not len(ib):continue
        n=n1[ia,k,None];p=n2[None,ib,k];L=A1[ia,k,None];M=A2[None,ib,k]
        correction=gamma_integral(a[k],b[k],n+p,L+M)-gamma_integral(a[k],b[k],n,L)-gamma_integral(a[k],b[k],p,M)
        shared[np.ix_(ia,ib)]-=correction
    output={'surface_shared':shared,'surface_independent':independent}
    for m,key in [('surface_mean','mean'),('surface_positive_only','positive')]:
        output[m]=np.array([s[key] for s in S1])[:,None]+np.array([s[key] for s in S2])[None,:]
    return {m:output[m] for m in methods}


def grid_exposure_tensor(O,D,s,weight,model,cfg):
    """Piecewise exact lengths and their local derivatives; reindex each call."""
    extent,h=cfg['hazard_extent_m'],cfg['hazard_voxel_m'];q=model['q']
    parallel=abs(D)<1e-12;safe=torch.where(parallel,torch.ones_like(D),D)
    aa=(-extent-O)/safe;bb=(extent-O)/safe
    low=torch.where(parallel,torch.full_like(aa,-1e10),torch.minimum(aa,bb)).max(1).values.clamp(min=0)
    high=torch.minimum(torch.where(parallel,torch.full_like(aa,1e10),torch.maximum(aa,bb)).min(1).values,s)
    valid=(high>low)&~(parallel&((O < -extent)|(O > extent))).any(1)
    O,D,low,high,weight=O[valid],D[valid],low[valid],high[valid],weight[valid]
    if not len(O):return torch.zeros(model['V'],dtype=DT)+O.sum()*0
    planes=-extent+torch.arange(1,q,dtype=DT)*h
    par=abs(D)<1e-12;safe=torch.where(par,torch.ones_like(D),D)
    cross=(planes[None,None,:]-O[:,:,None])/safe[:,:,None]
    cross=torch.where(par[:,:,None],high[:,None,None],cross).reshape(len(O),-1)
    times=torch.sort(torch.cat([low[:,None],torch.maximum(low[:,None],torch.minimum(high[:,None],cross)),high[:,None]],1),1).values
    length=torch.diff(times,dim=1);rr,cc=torch.where(length.detach()>1e-13)
    mid=(times[rr,cc]+times[rr,cc+1])/2
    xyz=torch.floor(((O[rr]+D[rr]*mid[:,None]).detach()+extent)/h).long().clamp(0,q-1)
    ids=(xyz[:,0]*q+xyz[:,1])*q+xyz[:,2]
    return torch.zeros(model['V'],dtype=DT).scatter_add(0,ids,length[rr,cc]*weight[rr]/h)


def surface_tensor(local,model,cfg):
    C=torch.as_tensor(model['center'],dtype=DT);N=torch.as_tensor(model['normal'],dtype=DT);Q=torch.as_tensor(model['precision'],dtype=DT)
    if not len(C):return torch.empty((len(local['P']),0),dtype=DT),torch.empty(0,dtype=DT)
    v=local['P'][:,None,:]-C;U=local['U']
    logK=.5*torch.log((U@N.T)**2+cfg['incidence_floor']**2)-np.log(np.sqrt(2*np.pi)*cfg['surface_epsilon_m'])-.5*torch.einsum('nfi,fij,nfj->nf',v,Q,v)
    O,D=local['O'],local['D'];v=O[:,None,:]-C
    aa=torch.einsum('ni,fij,nj->nf',D,Q,D);bb=torch.einsum('ni,fij,nfj->nf',D,Q,v);cc=torch.einsum('nfi,fij,nfj->nf',v,Q,v)
    lo=bb/torch.sqrt(aa);hi=lo+torch.sqrt(aa)*local['s'][:,None];root2=np.sqrt(2.)
    # erfc retains small tail probabilities that 0.5*(1+erf) loses.
    diff=torch.where(lo>0,.5*(torch.erfc(lo/root2)-torch.erfc(hi/root2)),torch.where(hi<0,.5*(torch.erfc(-hi/root2)-torch.erfc(-lo/root2)),.5*(torch.erf(hi/root2)-torch.erf(lo/root2))))
    incidence=torch.sqrt((D@N.T)**2+cfg['incidence_floor']**2)
    A=incidence/(cfg['surface_epsilon_m']*torch.sqrt(aa))*torch.exp(-.5*(cc-bb*bb/aa).clamp(min=0))*diff
    return logK,(A*local['weight'][:,None]).sum(0)


class PoseObjective:
    def __init__(self,data,slots,centers,model,a,b,cfg,method):
        self.model,self.cfg,self.method=model,cfg,method
        self.a=torch.tensor(a,dtype=DT);self.b=torch.tensor(b,dtype=DT);self.e=torch.digamma(self.a)-torch.log(self.b)
        self.frames=[]
        for d,s in zip(data,slots):
            ix=d['owner']==s
            self.frames.append({k:torch.tensor(v,dtype=DT) for k,v in dict(O=d['O']-centers[s],D=d['D'],s=d['s'],weight=d['weight'],P=d['PO'][ix]+d['PD'][ix]*d['ps'][ix,None]-centers[s],U=d['PD'][ix]).items()})

    def period(self,x,data):
        R=rotation(x[:3]);local=dict(data);local['O']=(data['O']-x[3:])@R;local['D']=data['D']@R;local['P']=(data['P']-x[3:])@R;local['U']=data['U']@R
        K,As=surface_tensor(local,self.model,self.cfg);V=self.model['V'];q=self.model['q'];h=self.cfg['hazard_voxel_m']
        xyz=torch.floor((local['P'].detach()+self.cfg['hazard_extent_m'])/h).long()
        if len(xyz) and not ((xyz>=0)&(xyz<q)).all():raise ValueError('Own hit outside workspace')
        ids=(xyz[:,0]*q+xyz[:,1])*q+xyz[:,2]
        logits=torch.cat([K+self.e[V:][None],(self.e[ids]-np.log(h))[:,None]],1);norm=torch.logsumexp(logits,1);resp=torch.softmax(logits,1)
        n=torch.cat([torch.zeros(V,dtype=DT).scatter_add(0,ids,resp[:,-1]),resp[:,:-1].sum(0)])
        c=norm.sum()-resp[:,:-1].sum(0)@self.e[V:]-resp[:,-1]@self.e[ids]
        if self.method in ('surface_mean','surface_positive_only'):
            lm=torch.log(self.a/self.b);positive=-torch.logsumexp(torch.cat([K+lm[V:][None],(lm[ids]-np.log(h))[:,None]],1),1).sum()
            if self.method=='surface_positive_only':return n,None,c,positive
        A=torch.cat([grid_exposure_tensor(local['O'],local['D'],local['s'],local['weight'],self.model,self.cfg),As])
        return n,A,c,(A@(self.a/self.b)+positive) if self.method=='surface_mean' else None

    def __call__(self,x,gradient=True):
        p=torch.tensor(x,dtype=DT,requires_grad=gradient)
        S=[self.period(p[6*t:6*t+6],d) for t,d in enumerate(self.frames)]
        def f(n,A):return torch.lgamma(self.a+n)-torch.lgamma(self.a)+self.a*torch.log(self.b)-(self.a+n)*torch.log(self.b+A)
        if self.method=='surface_shared':value=-f(S[0][0]+S[1][0],S[0][1]+S[1][1]).sum()-S[0][2]-S[1][2]
        elif self.method=='surface_independent':value=sum(-f(s[0],s[1]).sum()-s[2] for s in S)
        else:value=S[0][3]+S[1][3]
        if gradient:value.backward();return float(value.detach()),p.grad.detach().numpy()
        return float(value.detach())


def refine(objective,x,limits,cfg):
    from inspection_decisions.material_supported_continuous_alignment_v1 import project_parameters
    x=np.array(x,float);start=objective(x,False);value=start;trace=[start];gradnorm=[];evaluations=1
    for iteration in range(cfg['surface_refine_iterations']):
        _,g=objective(x,True);gradnorm.append(float(np.linalg.norm(g)))
        if not np.isfinite(g).all():raise FloatingPointError('Nonfinite geometry gradient')
        # Common bounded block-normalized step; actual objective tests all events.
        step=np.zeros(12)
        for j in (0,6):
            for off,scale in [(0,cfg['rotation_step_rad']),(3,cfg['translation_step_m'])]:
                v=g[j+off:j+off+3];step[j+off:j+off+3]=-v*scale/max(np.linalg.norm(v),1e-15)
        accepted=False
        for back in range(cfg['surface_backtracking']):
            p=project_parameters(x+step*.5**back,*limits);trial=objective(p,False);evaluations+=1
            if trial<value-1e-8:x,value=p,trial;trace.append(value);accepted=True;break
        if not accepted:break
    return x,dict(initial_cost=start,final_cost=value,cost_trace=trace,gradient_norms=gradnorm,evaluations=evaluations,
                  solver='autograd_surfaces_and_piecewise_voxel_exposure_actual_event_backtracking',global_optimum=False)
