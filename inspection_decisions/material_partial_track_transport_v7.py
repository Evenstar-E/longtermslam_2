"""Pose-conditioned censored partial material transport and joint tracks.

Assignment, multiway consistency and Gauss--Newton are standard tools. This
module is a testable task-model prototype, not a claim of original theory.
"""
from itertools import permutations,product
import numpy as np
from scipy.optimize import linear_sum_assignment
from inspection_decisions.textured_material_matching_v1 import apply
from inspection_decisions.material_correspondence_inference_v3 import inverse
from inspection_decisions.material_correspondence_mode_graph_v6 import skew,retract

PAIRS=[(0,1),(0,2),(1,2)]


def banks_from_features(ff):
    out=[]
    for e in range(3):
        row=[]
        for i in range(24):
            nodes={}
            for prefix in [k[:-6] for k in ff if k.endswith('_frame') and k.startswith(f'{e}_{i}_')]:
                for n,x,q in zip(ff[prefix+'_node'],ff[prefix+'_xyz'],ff[prefix+'_quality']):
                    n=int(n)
                    if n not in nodes or q>nodes[n][1]:nodes[n]=(x,float(q))
            keys=sorted(nodes);row.append(dict(node=np.array(keys,np.int64),xyz=np.array([nodes[n][0] for n in keys]).reshape(-1,3)))
        out.append(row)
    return out


def exposure(world,frames,slot,tolerance):
    N=len(world);hit=np.zeros(N,bool);gap=np.full(N,np.inf);normal=np.zeros((N,3));frame_index=np.full(N,-1,int)
    for fi,f in enumerate(frames):
        C=f['C'];pc=(world-C[:3,3])@C[:3,:3];z=pc[:,2];uv=pc[:,:2]/np.maximum(z[:,None],1e-9)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2];xy=np.rint(uv).astype(int);h,w=f['depth'].shape
        ok=(z>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);ii=np.flatnonzero(ok)
        if not len(ii):continue
        x,y=xy[ii].T;valid=f['depth_valid'][y,x];ii,x,y=ii[valid],x[valid],y[valid];dz=f['depth'][y,x]-z[ii];yes=(abs(dz)<=tolerance)&(f['anonymous_slot'][y,x]==slot);hit[ii[yes]]=True
        take=(dz>tolerance)&(dz-tolerance<gap[ii]);jj=ii[take];gap[jj]=dz[take]-tolerance;normal[jj]=C[:3,2];frame_index[jj]=fi
    active=np.isfinite(gap)&~hit;gap=np.where(active,gap,0.);normal[~active]=0.;frame_index[~active]=-1
    return dict(hit=hit,gap=gap,normal=normal,active=active,frame=frame_index)


def optional_assignment(gain):
    """Exact optional unit-capacity pair assignment, with neutral dummy nodes."""
    n,m=gain.shape
    if not n or not m:return np.zeros((0,2),int),0.
    aa=np.flatnonzero((gain>0).any(1));bb=np.flatnonzero((gain>0).any(0))
    if not len(aa) or not len(bb):return np.zeros((0,2),int),0.
    g=gain[np.ix_(aa,bb)];r,c=linear_sum_assignment(np.c_[-g,np.zeros((len(aa),len(aa)))])
    keep=c<len(bb);r,c=r[keep],c[keep];keep=g[r,c]>1e-12;pairs=np.c_[aa[r[keep]],bb[c[keep]]]
    return pairs,float(gain[pairs[:,0],pairs[:,1]].sum())


class PairTransport:
    def __init__(self,a,b,edge,frames_a,frames_b,slot_a,slot_b,cfg):
        self.a=a;self.b=b;self.fa=frames_a;self.fb=frames_b;self.ia=slot_a;self.ib=slot_b;self.cfg=cfg
        na={int(n):i for i,n in enumerate(a['node'])};nb={int(n):i for i,n in enumerate(b['node'])};self.conf=np.zeros((len(na),len(nb)))
        for n,m,w in zip(edge['source'],edge['target'],edge['weight']):
            i,j=na[int(n)],nb[int(m)];self.conf[i,j]=max(self.conf[i,j],float(w))

    def evidence(self,T):
        x=apply(T,self.a['xyz']);y=self.b['xyz'];reverse=apply(inverse(T),y);s=self.cfg['sigma_m'];tol=self.cfg['ray_tolerance_m']
        ea=exposure(x,self.fb,self.ib,tol);eb=exposure(reverse,self.fa,self.ia,tol)
        d2=np.maximum(0.,np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None]-2*x@y.T)
        # Confidence is a bounded appearance bonus, not a probability.
        cost=d2/(s*s)+1-self.conf
        u=ea['hit'].astype(float);v=eb['hit'].astype(float);gain=u[:,None]+v[None]-cost
        matched,value=optional_assignment(gain)
        free=.5*(np.log1p((ea['gap']/s)**2).mean() if len(x) else 0.)+.5*(np.log1p((eb['gap']/s)**2).mean() if len(y) else 0.)
        return dict(gain=gain,matched=matched,value=value,potential=float(u.sum()+v.sum()),free=float(free),exposure=(ea,eb))


def pair_gain(gains,e,i,f,j):
    if e>f:e,f,i,j=f,e,j,i
    return gains[{(0,1):0,(0,2):1,(1,2):2}[e,f]][i,j]


def partition_value(clusters,gains):
    value=0.
    for pe,(e,f) in enumerate(PAIRS):
        ok=(clusters[:,e]>=0)&(clusters[:,f]>=0);value+=gains[pe][clusters[ok,e],clusters[ok,f]].sum()
    return float(value)


def attach_epoch(clusters,epoch,count,gains):
    # Removing an epoch preserves all other material links and capacities.
    rest=np.array(clusters,copy=True).reshape(-1,3);rest[:,epoch]=-1;rest=rest[(rest>=0).any(1)];benefit=np.zeros((len(rest),count))
    for e in range(3):
        if e==epoch:continue
        ii=np.flatnonzero(rest[:,e]>=0)
        if not len(ii):continue
        pe={(0,1):0,(0,2):1,(1,2):2}[min(e,epoch),max(e,epoch)]
        benefit[ii]+=gains[pe][rest[ii,e],:] if e<epoch else gains[pe][:,rest[ii,e]].T
    pairs,_=optional_assignment(benefit);used=np.zeros(count,bool)
    rest[pairs[:,0],epoch]=pairs[:,1];used[pairs[:,1]]=True
    new=np.full((int((~used).sum()),3),-1,int);new[:,epoch]=np.flatnonzero(~used)
    return np.r_[rest,new]


def material_partition(gains,sweeps=0):
    n0,n1=gains[0].shape;n2=gains[1].shape[1]
    clusters=np.full((n0,3),-1,int);clusters[:,0]=np.arange(n0)
    clusters=attach_epoch(clusters,1,n1,gains);clusters=attach_epoch(clusters,2,n2,gains);path=[partition_value(clusters,gains)]
    for _ in range(sweeps):
        before=path[-1]
        for e,n in enumerate([n0,n1,n2]):
            clusters=attach_epoch(clusters,e,n,gains);value=partition_value(clusters,gains);assert value>=path[-1]-1e-8;path.append(value)
        if path[-1]-before<1e-10:break
    return clusters,np.array(path)


def joint_values(evidence,cfg,return_states=False):
    weights=[1.,1.,.5];gains=[w*r['gain'] for w,r in zip(weights,evidence)];Z=sum(w*r['potential'] for w,r in zip(weights,evidence))+cfg['support_regularizer'];free=sum(w*r['free'] for w,r in zip(weights,evidence))/2.5
    independent=sum(w*r['value'] for w,r in zip(weights,evidence));memory,mpath=material_partition(gains,0);shared,spath=material_partition(gains,cfg['track_sweeps'])
    assert spath[-1]<=independent+1e-8
    values=1-np.array([independent,mpath[-1],spath[-1]])/Z+cfg['free_weight']*free
    if return_states:return values,dict(gains=gains,Z=Z,free=free,memory=memory,shared=shared,memory_path=mpath,shared_path=spath)
    return values


def selected_edges(evidence,state,mode,banks):
    result=[]
    for pe,(e,f) in enumerate(PAIRS):
        if mode=='visibility_pair':pairs=evidence[pe]['matched']
        else:
            clusters=state['memory' if mode=='history_union' else 'shared'];ok=(clusters[:,e]>=0)&(clusters[:,f]>=0);pairs=clusters[ok][:,[e,f]]
        result.append(dict(x=banks[e]['xyz'][pairs[:,0]],y=banks[f]['xyz'][pairs[:,1]],source=pairs[:,0],target=pairs[:,1],weight=np.ones(len(pairs))))
    return result


def transformed_jacobian(G,h,f,x):
    R=G[f][:3,:3]@G[h][:3,:3].T;z=apply(G[f]@inverse(G[h]),x);J=np.zeros((len(x),3,12))
    if f:J[:,:,(f-1)*6:(f-1)*6+3]=-skew(z);J[:,:,(f-1)*6+3:f*6]=np.eye(3)
    if h:J[:,:,(h-1)*6:(h-1)*6+3]+=R@skew(x);J[:,:,(h-1)*6+3:h*6]-=R
    return z,J


def fit(T,banks,pairs,cfg,mode):
    index={'visibility_pair':0,'history_union':1,'shared_transport':2}[mode];T=np.array(T);path=[];snapshots=[]
    def evaluate(T,states=False):
        ev=[pairs[0].evidence(T[0]),pairs[1].evidence(T[1]),pairs[2].evidence(T[1]@inverse(T[0]))];values,state=joint_values(ev,cfg,True)
        return float(values[index]),ev,state
    current,ev,state=evaluate(T);path.append(current)
    for _ in range(cfg['pose_steps']):
        G=np.r_[np.eye(4)[None],T];edges=selected_edges(ev,state,mode,banks);AA=[];bb=[];s=cfg['sigma_m']
        for pe,((h,f),r,w) in enumerate(zip(PAIRS,edges,[1.,1.,.5])):
            z,J=transformed_jacobian(G,h,f,r['x']);factor=np.sqrt(w/state['Z'])/s;AA.append((J*factor).reshape(-1,12));bb.append(((z-r['y'])*factor).ravel())
            for direction,(a,b) in enumerate([(h,f),(f,h)]):
                x=banks[a]['xyz'];z,J=transformed_jacobian(G,a,b,x);info=ev[pe]['exposure'][direction];active=info['active'];d=info['gap'][active];D=-np.einsum('ni,nij->nj',info['normal'][active],J[active]);fac=np.sqrt(cfg['free_weight']*w/(2.5*2*max(len(x),1)))/np.sqrt(s*s+d*d);AA.append(D*fac[:,None]);bb.append(d*fac)
        A=np.concatenate(AA);b=np.concatenate(bb)
        if not len(b):break
        delta=np.linalg.lstsq(np.r_[A,np.eye(12)*.001],np.r_[-b,np.zeros(12)],rcond=None)[0];delta*=min(1,cfg['step_norm']/max(np.linalg.norm(delta),1e-12));accepted=False
        for alpha in [1.,.5,.25,.125]:
            candidate=retract(T,alpha*delta);value,ne,ns=evaluate(candidate)
            if value<current-1e-12:T=candidate;current=value;ev,state=ne,ns;accepted=True;break
        path.append(current)
        if not accepted:break
    return dict(T=T,path=np.array(path),clusters=state['memory' if mode=='history_union' else 'shared'] if mode!='visibility_pair' else np.zeros((0,3),int),Z=state['Z'])


def geometry_starts(a,b):
    """Standard PCA signed-permutation hypotheses; shared by all methods."""
    _,A=np.linalg.eigh(np.cov(a.T));_,B=np.linalg.eigh(np.cov(b.T));out=[]
    for perm in permutations(range(3)):
        for sign in product([-1.,1.],repeat=3):
            S=np.eye(3)[:,perm]*sign;R=B@S@A.T
            if np.linalg.det(R)<0:continue
            rotated=a@R.T;ca=(rotated.min(0)+rotated.max(0))/2;cb=(b.min(0)+b.max(0))/2;T=np.eye(4);T[:3,:3]=R;T[:3,3]=cb-ca;out.append(T)
    return np.array(out)
