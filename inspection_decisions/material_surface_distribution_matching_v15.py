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
    """Every native target pixel competes; no coarse-pose/3-anchor restriction."""
    tree=b['_tree'] if '_tree' in b else cKDTree(b['xyz']);b['_tree']=tree;points=[];means=[];cov=[];mass=[];sources=[];cosines=[]
    for start in range(0,len(a['xyz']),64):
        ix=np.arange(start,min(start+64,len(a['xyz'])));sim=np.asarray(a['desc'][ix],np.float32)@np.asarray(b['desc'],np.float32).T;kernel=np.exp((sim-1)/.12);den=kernel@b['area'];work=sim.copy()
        for _ in range(cfg['modes_per_source']):
            peak=work.argmax(1);usable=np.isfinite(work[np.arange(len(ix)),peak])
            for row,i in enumerate(ix):
                if not usable[row]:continue
                j=int(peak[row]);near=np.array(tree.query_ball_point(b['xyz'][j],cfg['mode_radius_m']),int);w=kernel[row,near]*b['area'][near];total=w.sum();w/=max(total,1e-30);mu=w@b['xyz'][near];delta=b['xyz'][near]-mu;C=np.einsum('n,ni,nj->ij',w,delta,delta)
                # Keep actual native peak for point control; same region mean
                # and covariance are available to both proposal mechanisms.
                points.append(b['xyz'][j]);means.append(mu);cov.append(C);mass.append(total/max(den[row],1e-30));sources.append(i);cosines.append(sim[row,j])
            d2=np.sum((b['xyz'][None]-b['xyz'][peak,None])**2,-1);work[d2<cfg['mode_separation_m']**2]=-np.inf
    return dict(source=np.array(sources,int),point=np.array(points),mean=np.array(means),cov=np.array(cov),mass=np.array(mass),cosine=np.array(cosines))


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


class LearnedPair:
    def __init__(self,pair,cfg):self.p=pair;self.cfg=cfg
    def feature_values(self,T):
        p=self.p;values=[]
        for a,b,tree,U in [(p.a,p.db,p.tb,T),(p.b,p.da,p.ta,inverse(T))]:
            x=apply(U,a['xyz']);d,j=tree.query(x,k=4);sim=np.sum(a['desc'][:,None]*b['desc'][j],axis=-1);fusion=np.min(np.minimum(d/.06,1)**2+.5*(1-sim),axis=1).mean()
            # Local spatial quadrature. All points inside4sigma participate;
            # no fixed-neighbor truncation that changes with pixel density.
            near=tree.query_ball_point(x,4*self.cfg['marginal_sigma_m']);counts=np.array([len(h) for h in near]);ids=np.repeat(np.arange(len(x)),counts);jj=np.concatenate([np.asarray(h,int) for h in near]) if counts.sum() else np.array([],int);num=np.zeros(len(x));den=np.zeros(len(x))
            if len(jj):
                g=np.exp(-.5*np.sum((x[ids]-b['xyz'][jj])**2,axis=1)/self.cfg['marginal_sigma_m']**2);k=np.exp((np.sum(a['desc'][ids]*b['desc'][jj],axis=1)-1)/.12);weight=g*b['area'][jj];np.add.at(num,ids,weight*k);np.add.at(den,ids,weight)
            ratio=np.ones(len(x));np.divide(num,den*a['normalizer'],out=ratio,where=den>1e-20);support=np.exp(-.5*(d[:,0]/self.cfg['marginal_sigma_m'])**2);evidence=1-support+support*ratio;marginal=-np.log(np.maximum(evidence,1e-12)).mean();values.append((fusion,marginal))
        return np.mean(values,axis=0)
