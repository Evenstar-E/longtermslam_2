"""Metric measured texture charts and explicit partial material correspondences.

Local frames, circular correlation, optional assignment and GN are known tools.
The masked/multimodal material-frame experiment is a hypothesis, not novelty.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from inspection_decisions.material_inverse_plane_control_v11 import project
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from inspection_decisions.material_partial_track_transport_v7 import (
    optional_assignment,material_partition,selected_edges,transformed_jacobian)
from inspection_decisions.material_correspondence_mode_graph_v6 import retract

PAIRS=[(0,1),(0,2),(1,2)]


def farthest_indices(x,n):
    if len(x)<=n:return np.arange(len(x))
    selected=[];d=np.full(len(x),np.inf);h=0
    for _ in range(n):
        selected.append(h);d=np.minimum(d,np.sum((x-x[h])**2,1));d[selected]=-1;h=int(d.argmax())
    return np.array(selected)


def chart_basis(x,points,camera,cfg):
    if len(points)<8:return None
    centered=points-points.mean(0);_,s,V=np.linalg.svd(centered,full_matrices=False)
    if s[1]<1e-5 or s[0]<1e-5:return None
    normal=V[-1]
    if np.sqrt(np.mean((centered@normal)**2))>cfg['chart_plane_rmse_m']:return None
    if normal@(camera[:3,3]-x)<0:normal=-normal
    tangent=camera[:3,0]-normal*(normal@camera[:3,0])
    if np.linalg.norm(tangent)<1e-5:tangent=camera[:3,1]-normal*(normal@camera[:3,1])
    tangent/=np.linalg.norm(tangent);return np.stack([tangent,np.cross(normal,tangent),normal],1)


def whiten_samples(rgb,mask):
    weight=mask[...,None];den=np.maximum(mask.sum((-2,-1)),1)[...,None,None,None]
    mean=(rgb*weight).sum((-3,-2),keepdims=True)/den
    centered=(rgb-mean)*weight;energy=np.sqrt(np.sum(centered**2,axis=(-3,-2,-1),keepdims=True))
    return centered/np.maximum(energy,1e-8),energy[...,0,0,0]


def build_charts(bank,frames,slot,cfg):
    xyz=bank['xyz'];tree=cKDTree(xyz);eligible=[];bases=[]
    for h,x in enumerate(xyz):
        f=frames[int(bank['frame'][h])];basis=chart_basis(x,xyz[tree.query_ball_point(x,cfg['chart_neighbor_radius_m'])],f['C'],cfg)
        if basis is not None:eligible.append(h);bases.append(basis)
    # Keep a well-defined fallback for genuinely sparse/no-plane slots.
    if not eligible:
        eligible=list(range(len(xyz)));bases=[]
        for h in eligible:
            n=bank['normal'][h];n=n/max(np.linalg.norm(n),1e-12);t=np.eye(3)[np.argmin(abs(n))];t-=n*(t@n);t/=np.linalg.norm(t);bases.append(np.stack([t,np.cross(n,t),n],1))
    take=farthest_indices(xyz[eligible],cfg['chart_nodes']);idx=np.array(eligible)[take];basis=np.array(bases)[take]
    angle=np.arange(cfg['chart_angles'])*2*np.pi/cfg['chart_angles'];r=np.array(cfg['chart_radii_m'])
    local=np.stack([r[:,None]*np.cos(angle),r[:,None]*np.sin(angle)],-1)
    world=xyz[idx,None,None,:]+np.einsum('nij,raj->nrai',basis[:,:,:2],local)
    rgb=np.zeros(world.shape);mask=np.zeros(world.shape[:-1],bool)
    for fi in np.unique(bank['frame'][idx]):
        ix=np.flatnonzero(bank['frame'][idx]==fi);q=project(world[ix],frames[int(fi)],slot)
        mask[ix]=q['surface']&(abs(q['gap'])<=cfg['chart_self_tolerance_m'])&(np.sum(q['normal']*basis[ix,None,None,:,2],axis=-1)>.7);rgb[ix]=q['rgb']
    texture,energy=whiten_samples(rgb,mask);spectrum=abs(np.fft.rfft(texture,axis=2)).reshape(len(idx),-1);spectrum/=np.maximum(np.linalg.norm(spectrum,axis=1,keepdims=True),1e-8)
    fpfh=bank['fpfh'][:,idx].T;fpfh/=np.maximum(np.linalg.norm(fpfh,axis=1,keepdims=True),1e-8)
    return dict(xyz=xyz[idx],basis=basis,texture=texture,texture_energy=energy,mask=mask,rgb=rgb,ring_xyz=world,uv=bank['uv'][idx],frame=bank['frame'][idx],center_index=idx,spectrum=spectrum,fpfh=fpfh)


def phase_distribution(a,b,cfg):
    # Fixed observed patch energy denominators: masking cannot improve NCC by
    # silently shrinking a pose-dependent denominator. Missing samples are zero.
    A=a['texture'].reshape(len(a['xyz']),-1);corr=[]
    for h in range(cfg['chart_angles']):corr.append(A@np.roll(b['texture'],-h,axis=2).reshape(len(b['xyz']),-1).T)
    corr=np.clip(np.stack(corr,-1),-1,1);logits=corr/cfg['phase_temperature'];logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True)
    reliability=1+np.sum(p*np.log(np.maximum(p,1e-12)),axis=-1)/np.log(cfg['chart_angles'])
    return corr,p,np.clip(reliability,0,1)


def rotation_z(theta):
    c,s=np.cos(theta),np.sin(theta);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def phase_angle(correlation,h):
    L=len(correlation);left,mid,right=correlation[(h-1)%L],correlation[h],correlation[(h+1)%L]
    den=left-2*mid+right;delta=np.clip(.5*(left-right)/den,-.5,.5) if den<-1e-8 else 0.
    return 2*np.pi*(h+delta)/L


class ChartPair:
    def __init__(self,a,b,fa,fb,ia,ib,cfg):
        self.a=a;self.b=b;self.fa=fa;self.fb=fb;self.ia=ia;self.ib=ib;self.cfg=cfg
        self.corr,self.prob,self.reliability=phase_distribution(a,b,cfg)
        geom=np.maximum(0.,1-a['fpfh']@b['fpfh'].T);spec=np.maximum(0.,1-a['spectrum']@b['spectrum'].T)
        self.descriptor=.5*geom+.5*spec;self.texture_distance=1-self.corr.max(-1)

    def exposure(self,world,frames,slot):
        hit=np.zeros(len(world),bool);gap=np.full(len(world),np.inf);normal=np.zeros_like(world)
        for f in frames:
            r=project(world,f,slot);hit|=r['surface']&(abs(r['gap'])<=self.cfg['ray_tolerance_m']);free=r['valid']&(r['gap']>self.cfg['ray_tolerance_m']);take=free&(r['gap']-self.cfg['ray_tolerance_m']<gap);gap[take]=r['gap'][take]-self.cfg['ray_tolerance_m'];normal[take]=-r['normal'][take]
        active=np.isfinite(gap)&~hit;gap=np.where(active,gap,0.);normal[~active]=0
        return dict(hit=hit,gap=gap,normal=normal,active=active)

    def evidence(self,T,oriented):
        a,b=self.a,self.b;x=apply(T,a['xyz']);y=b['xyz'];reverse=apply(inverse(T),y);s=self.cfg['match_sigma_m']
        if not hasattr(self,'last_T') or not np.array_equal(T,self.last_T):
            ea=self.exposure(x,self.fb,self.ib);eb=self.exposure(reverse,self.fa,self.ia)
            d2=np.maximum(0.,(x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T)
            base=d2/s**2+self.descriptor/self.cfg['correspondence_cost_scale'];free=.5*np.log1p((ea['gap']/s)**2).mean()+.5*np.log1p((eb['gap']/s)**2).mean()
            self.last_T=T.copy();self.last_base=(ea,eb,d2,base,free)
        ea,eb,d2,base,free=self.last_base;cost=base.copy()
        if oriented:
            # Each frame is a measured chart. Keep the circular distribution,
            # do not replace multiple texture peaks by a mean orientation.
            R=np.einsum('nji,jk,mkl->nmil',b['basis'],T[:3,:3],a['basis'])
            # R indexes target first/source second; transpose to source/target.
            R=R.transpose(1,0,2,3);theta=np.arctan2(R[:,:,1,0],R[:,:,0,0]);u=np.mod(theta,2*np.pi)*self.cfg['chart_angles']/(2*np.pi);lo=np.floor(u).astype(int);frac=u-lo
            p0=np.take_along_axis(self.prob,lo[...,None],axis=-1)[...,0];p1=np.take_along_axis(self.prob,((lo+1)%self.cfg['chart_angles'])[...,None],axis=-1)[...,0];p=(1-frac)*p0+frac*p1
            phase=-np.log(np.maximum(p,1e-12))/np.log(self.cfg['chart_angles']);normal=np.clip(R[:,:,2,2],-1,1)
            cost+=self.cfg['normal_cost_weight']*(1-normal)
            if oriented==2:cost+=self.cfg['phase_cost_weight']*self.reliability*phase
        gain=ea['hit'][:,None].astype(float)+eb['hit'][None,:].astype(float)-cost
        matched,value=optional_assignment(gain)
        return dict(gain=gain,matched=matched,value=value,potential=float(len(x)+len(y)),free=float(free),exposure=(ea,eb))

    def fusion(self,T):
        self.evidence(T,0);d2=self.last_base[2];ia=d2.argmin(1);ib=d2.argmin(0);n,m=d2.shape
        return .5*((np.minimum(np.sqrt(d2[np.arange(n),ia])/self.cfg['match_sigma_m'],1)+self.descriptor[np.arange(n),ia]).mean()+(np.minimum(np.sqrt(d2[ib,np.arange(m)])/self.cfg['match_sigma_m'],1)+self.descriptor[ib,np.arange(m)]).mean())

    def proposals(self,seed):
        rng=np.random.default_rng(seed);n,m=self.descriptor.shape
        # Explicit finite proposal budget; scoring itself uses ALL n*m edges.
        edge=np.array([(i,j) for i in range(n) for j in np.argsort(self.descriptor[i])[:self.cfg['proposal_edges_per_source']]],int)
        ia,ib=edge.T;A=self.a['xyz'][ia];B=self.b['xyz'][ib];ordinary=[]
        for _ in range(self.cfg['ordinary_ransac_trials']):
            h=rng.choice(len(edge),3,replace=False);T=rigid(A[h],B[h])
            if T is not None:ordinary.append(T)
        oriented=[]
        for i,j in edge:
            modes=np.argsort(-self.corr[i,j])[:2] if self.reliability[i,j]>.02 else np.arange(0,self.cfg['chart_angles'],2)
            for h in modes:
                angle=phase_angle(self.corr[i,j],int(h));R=self.b['basis'][j]@rotation_z(angle)@self.a['basis'][i].T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=self.b['xyz'][j]-R@self.a['xyz'][i];oriented.append(T)
        def rank(TT):
            if not TT:return np.empty((0,4,4))
            TT=np.array(TT);values=[]
            for begin in range(0,len(TT),64):
                U=TT[begin:begin+64];x=np.einsum('bij,nj->bni',U[:,:3,:3],self.a['xyz'])+U[:,None,:3,3];dist,index=cKDTree(self.b['xyz']).query(x)
                appearance=self.descriptor[np.arange(n)[None],index];values.extend((np.exp(-.5*(dist/self.cfg['match_sigma_m'])**2)*np.exp(-appearance)).mean(1))
            return TT[np.argsort(-np.array(values),kind='stable')[:self.cfg['proposal_keep']]]
        return rank(ordinary),rank(oriented),dict(ordinary_trials=len(ordinary),frame_trials=len(oriented),all_correspondence_edges=n*m,proposal_edges=len(edge))


def triple_values(evidence,cfg,shared=False):
    weights=[1.,1.,.5];gains=[w*r['gain'] for w,r in zip(weights,evidence)];Z=sum(w*r['potential'] for w,r in zip(weights,evidence))
    independent=sum(w*r['value'] for w,r in zip(weights,evidence))
    if shared:
        clusters,path=material_partition(gains,cfg['track_sweeps']);value=path[-1]
    else:clusters=np.empty((0,3),int);value=independent
    free=sum(w*r['free'] for w,r in zip(weights,evidence))/2.5
    return 1-value/Z+cfg['free_weight']*free,dict(gains=gains,Z=Z,free=free,shared=clusters,memory=clusters,independent_gain=independent)


class ChartTriple:
    def __init__(self,pairs,cfg):self.pairs=pairs;self.cfg=cfg
    def evaluate(self,T,states=False):
        U=[T[0],T[1],T[1]@inverse(T[0])];values=[];evs=[];sts=[]
        for oriented in [0,1,2]:
            ev=[p.evidence(u,oriented) for p,u in zip(self.pairs,U)]
            for shared in [False,True]:
                value,state=triple_values(ev,self.cfg,shared);values.append(value);evs.append(ev);sts.append(state)
        # independent/shared point, independent/shared normal, ind/shared phase.
        return (np.array(values),evs,sts) if states else np.array(values)
    def score(self,T):return np.array([self.evaluate(t) for t in T])


def fit_correspondences(scorer,T,cfg,column):
    T=np.array(T);values,evs,states=scorer.evaluate(T,True);value=float(values[column]);path=[value]
    banks=[scorer.pairs[0].a,scorer.pairs[0].b,scorer.pairs[1].b]
    for _ in range(cfg['correspondence_pose_steps']):
        mode='shared' if column%2 else 'visibility_pair';state=states[column];ev=evs[column]
        edges=selected_edges(ev,state,mode,banks);G=np.r_[np.eye(4)[None],T];AA=[];rr=[]
        for (h,f),r,w in zip(PAIRS,edges,[1.,1.,.5]):
            if not len(r['x']):continue
            z,J=transformed_jacobian(G,h,f,r['x']);fac=np.sqrt(w/max(state['Z'],1))/cfg['match_sigma_m'];AA.append((J*fac).reshape(-1,12));rr.append(((z-r['y'])*fac).ravel())
        if not AA:break
        A=np.concatenate(AA);b=np.concatenate(rr);d=np.linalg.lstsq(np.r_[A,.01*np.eye(12)],np.r_[-b,np.zeros(12)],rcond=None)[0];d*=min(1,.05/max(np.linalg.norm(d),1e-12))
        accepted=False
        for alpha in [1,.5,.25,.125]:
            U=retract(T,d*alpha);vv,ee,ss=scorer.evaluate(U,True)
            if vv[column]<value-1e-12:T=U;values=vv;evs=ee;states=ss;value=float(vv[column]);accepted=True;break
        path.append(value)
        if not accepted:break
    return T,np.array(path)
