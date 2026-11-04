"""Dense learned surface modes and uncertainty-aware global correspondence.

No truth or data-file reads. V3's trained RGBD encoder remains unchanged.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from inspection_decisions.material_joint_descriptor_v1 import metric_patches
from inspection_decisions.textured_material_matching_v1 import rigid,apply,world_points
from inspection_decisions.material_surface_patch_registration_v9 import inverse


@torch.inference_mode()
def encode_record(model,record,frames,feature_cfg,cfg):
    out=np.zeros((len(record['xyz']),64),np.float32)
    for fi in np.unique(record['frame']):
        ix=np.flatnonzero(record['frame']==fi);frame=frames[int(fi)]
        for start in range(0,len(ix),cfg['encode_batch']):
            h=ix[start:start+cfg['encode_batch']];patch,xyz=metric_patches(frame,record['uv'][h],feature_cfg)
            assert np.max(abs(xyz-record['xyz'][h]))<1e-7
            out[h]=model.encode(torch.from_numpy(patch)).numpy()
    return out


def native_area(frame,slot,edge_limit=.12):
    """One third of witnessed triangle area per vertex; isolated native fallback.

This is a measured surface quadrature proxy, NOT exact area of unseen surfaces.
"""
    H,W=frame['depth'].shape;yy,xx=np.indices((H,W));p=world_points(frame,np.c_[xx.ravel(),yy.ravel()]).reshape(H,W,3);valid=frame['depth_valid']&(frame['anonymous_slot']==slot);area=np.zeros((H,W))
    for corners in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        points=[];good=[];slices=[]
        for y,x in corners:
            sl=(slice(y,H-1+y),slice(x,W-1+x));slices.append(sl);points.append(p[sl]);good.append(valid[sl])
        a,b,c=points;ok=np.logical_and.reduce(good)&(np.maximum.reduce([np.linalg.norm(a-b,axis=-1),np.linalg.norm(a-c,axis=-1),np.linalg.norm(b-c,axis=-1)])<=edge_limit)
        mass=np.linalg.norm(np.cross(b-a,c-a),axis=-1)/6*ok
        for sl in slices:area[sl]+=mass
    fallback=frame['depth']**2/(frame['K'][0,0]*frame['K'][1,1]);return np.where(valid,np.where(area>0,area,fallback),0.)


def add_area(record,frames,slot):
    area=np.zeros(len(record['xyz']))
    for fi in np.unique(record['frame']):
        ix=np.flatnonzero(record['frame']==fi);f=frames[int(fi)];A=native_area(f,slot);uv=record['uv'][ix].astype(int);area[ix]=A[uv[:,1],uv[:,0]]
    # Fixed average over observed views; normalized per slot afterwards.
    assert (area>0).all();record['area']=area/area.sum();return record


def dense_modes(a,b,cfg):
    tree=b['_tree'] if '_tree' in b else cKDTree(b['xyz']);b['_tree']=tree;points=[];means=[];cov=[];prior_means=[];prior_cov=[];mass=[];prior=[];sources=[];cosines=[];normalizer=np.zeros(len(a['xyz']))
    for start in range(0,len(a['xyz']),64):
        ix=np.arange(start,min(start+64,len(a['xyz'])));sim=np.asarray(a['desc'][ix],np.float32)@np.asarray(b['desc'],np.float32).T;kernel=np.exp((sim-1)/.12);den=kernel@b['area'];normalizer[ix]=den;work=sim.copy();peaks=[];usable=[]
        for _ in range(cfg['modes_per_source']):
            peak=work.argmax(1);ok=np.isfinite(work[np.arange(len(ix)),peak]);peaks.append(peak);usable.append(ok);d2=np.sum((b['xyz'][None]-b['xyz'][peak,None])**2,-1);work[d2<cfg['mode_separation_m']**2]=-np.inf
        peaks=np.array(peaks).T;usable=np.array(usable).T
        for row,i in enumerate(ix):
            peak=peaks[row,usable[row]];centers=b['xyz'][peak];union=np.unique(np.concatenate([np.asarray(h,int) for h in tree.query_ball_point(centers,cfg['mode_radius_m'])]));distance=np.linalg.norm(b['xyz'][union,None]-centers[None],axis=-1);assign=distance.argmin(1)
            for mode,j in enumerate(peak):
                near=union[(assign==mode)&(distance[:,mode]<=cfg['mode_radius_m'])];w=kernel[row,near]*b['area'][near];total=w.sum();w/=max(total,1e-30);mu=w@b['xyz'][near];delta=b['xyz'][near]-mu;C=np.einsum('n,ni,nj->ij',w,delta,delta)
                w0=b['area'][near]/b['area'][near].sum();mu0=w0@b['xyz'][near];d0=b['xyz'][near]-mu0;C0=np.einsum('n,ni,nj->ij',w0,d0,d0)
                points.append(b['xyz'][j]);means.append(mu);cov.append(C);prior_means.append(mu0);prior_cov.append(C0);mass.append(total/max(den[row],1e-30));prior.append(b['area'][near].sum());sources.append(i);cosines.append(sim[row,j])
    result=dict(source=np.array(sources,int),point=np.array(points),mean=np.array(means),cov=np.array(cov),prior_mean=np.array(prior_means),prior_cov=np.array(prior_cov),mass=np.array(mass),prior=np.array(prior),cosine=np.array(cosines),slot_normalizer=normalizer)
    for key in ['mass','prior']:assert (np.bincount(result['source'],weights=result[key],minlength=len(a['xyz']))<=1+1e-6).all(),key
    return result


def compatibility(a,modes,cfg,regional):
    x=a['xyz'][modes['source']];y=modes['mean'] if regional else modes['point'];dx=np.linalg.norm(x[:,None]-x[None],axis=-1);delta=y[:,None]-y[None];dy=np.linalg.norm(delta,axis=-1);sigma2=cfg['compatibility_sigma_m']**2
    variance=np.zeros_like(dy)
    if regional:
        direction=delta/np.maximum(dy[...,None],1e-12);variance=np.einsum('abi,abij,abj->ab',direction,modes['cov'][:,None]+modes['cov'][None],direction)
    # Gaussian convolution includes its concentration factor: broad regions
    # cannot gain arbitrary compatibility merely by inflating uncertainty.
    C=np.sqrt(sigma2/(sigma2+variance))*np.exp(-.5*(dx-dy)**2/(sigma2+variance));valid=(modes['source'][:,None]!=modes['source'][None])&(dy>.015);C*=valid
    return C.astype(np.float32),y


def global_proposals(a,modes,cfg,regional):
    C,y=compatibility(a,modes,cfg,regional);C2=(C@C)*C;v=np.sqrt(np.maximum(modes['mass'],1e-12)).astype(np.float32)
    for _ in range(8):v=C2@v;v/=max(np.linalg.norm(v),1e-12)
    seeds=[];poses=[];x=a['xyz'][modes['source']]
    for h in np.argsort(-v):
        if seeds and np.min(np.linalg.norm(x[seeds]-x[h],axis=1))<.04:continue
        seeds.append(int(h));weights=C2[h]*v;order=np.argsort(-weights);chosen=[];seen=set()
        for k in order:
            source=int(modes['source'][k])
            if source in seen:continue
            seen.add(source);chosen.append(int(k))
            if len(chosen)>=cfg['spectral_consensus']:break
        hix=np.array(chosen);T=rigid(x[hix],y[hix],weights[hix]+1e-12)
        if T is not None:poses.append(T)
        if len(seeds)>=cfg['spectral_seeds']:break
    return np.array(poses).reshape(-1,4,4)


class RegionPair:
    def __init__(self,pair,forward,backward,cfg):
        self.p=pair;self.modes=[forward,backward];self.cfg=cfg;self.precision=[];self.prefactor=[]
        s2=cfg['marginal_sigma_m']**2
        for m in self.modes:
            S=s2*np.eye(3)+np.stack([m['cov'],m['prior_cov']]);self.precision.append(np.linalg.inv(S));self.prefactor.append(s2**1.5/np.sqrt(np.linalg.det(S)))
    def one_mixture(self,x,a,m,precision,prefactor,kind):
        src=m['source'];target=np.stack([m['point'],m['point']]) if kind==0 else np.stack([m['mean'],m['prior_mean']]);r=x[src][None]-target;s2=self.cfg['marginal_sigma_m']**2
        if kind==2:K=prefactor*np.exp(-.5*np.einsum('ani,anij,anj->an',r,precision,r))
        else:K=np.exp(-.5*np.sum(r*r,axis=2)/s2)
        n=len(x);post=m['mass'];prior=m['prior'];remaining=np.maximum(0,1-np.bincount(src,weights=post,minlength=n));base=np.maximum(0,1-np.bincount(src,weights=prior,minlength=n));u=self.cfg['outlier_kernel'];num=remaining*u+np.bincount(src,weights=post*K[0],minlength=n);den=base*u+np.bincount(src,weights=prior*K[1],minlength=n)
        ratio=num/np.maximum(den,1e-20)*m['slot_normalizer']/a['normalizer'];return -np.log(np.maximum(ratio,1e-12)).mean()
    def values(self,T):
        p=self.p;geo,rgb,_=p.nearest(T);forward=apply(T,p.a['xyz']);backward=apply(inverse(T),p.b['xyz']);fixed,detail=p.projected(T,False);free=fixed-1+.5*(detail[0]['kernel'].mean()+detail[1]['kernel'].mean());fusion=[];mixture=np.zeros(3)
        for h,(a,b,tree,x) in enumerate([(p.a,p.db,p.tb,forward),(p.b,p.da,p.ta,backward)]):
            d,j=tree.query(x,k=4);sim=np.sum(a['desc'][:,None]*b['desc'][j],axis=-1);fusion.append(np.min(np.minimum(d/.06,1)**2+.5*(1-sim),axis=1).mean())
            for kind in range(3):mixture[kind]+=.5*self.one_mixture(x,a,self.modes[h],self.precision[h],self.prefactor[h],kind)
        return np.r_[geo,rgb,np.mean(fusion)+free,geo+free+self.cfg['marginal_weight']*mixture]
