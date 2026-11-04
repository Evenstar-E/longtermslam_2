"""Observed native material tracks and discrete correspondence-mode proposals.

Cycle consistency, compatibility graphs and IRLS are standard controls, not an
originality claim. No private labels, exact object poses, or catalogue axes.
"""
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.textured_material_matching_v1 import rigid,apply
from inspection_decisions.material_correspondence_inference_v3 import inverse


def budget_edges(record,cfg):
    order=np.argsort(-record['weight'],kind='stable');used=set();ns=defaultdict(int);nt=defaultdict(int);keep=[]
    for h in order:
        a,b=int(record['source'][h]),int(record['target'][h])
        if (a,b) in used or ns[a]>=2 or nt[b]>=2:continue
        used.add((a,b));ns[a]+=1;nt[b]+=1;keep.append(h)
        if len(keep)>=cfg['edges_per_pair']:break
    return {k:np.asarray(v)[keep] for k,v in record.items()}


def unpack_edges(packed,cfg):
    out=[]
    for pe in range(3):
        rows=[];off=packed[f'{pe}_offsets']
        for q in range(576):
            a,b=int(off[q]),int(off[q+1]);rows.append(budget_edges({k:packed[f'{pe}_{k}'][a:b] for k in ['x','y','source','target','weight']},cfg))
        out.append(rows)
    return out


def distances(x):return np.linalg.norm(x[:,None]-x[None],axis=2)


def compatible(x,y,cfg):
    dx,dy=distances(x),distances(y);C=np.maximum(0.,1-((dx-dy)/cfg['geometry_tolerance_m'])**2)
    C[(dx<cfg['exclusion_m'])|(dy<cfg['exclusion_m'])]=0
    return C


def cliques(C,weight,count=4):
    if len(C)<3:return []
    C2=C*(C@C)/len(C);v=np.maximum(weight,1e-8);v/=max(np.linalg.norm(v),1e-12)
    for _ in range(12):v=C2@v;v/=max(np.linalg.norm(v),1e-12)
    found=[]
    for seed in np.argsort(-v)[:min(16,len(v))]:
        order=np.argsort(-(C2[seed]*v));chosen=[int(seed)]
        for k in order:
            if k==seed or min(C[int(k),chosen])<.2:continue
            chosen.append(int(k))
            if len(chosen)>=32:break
        if len(chosen)<3:continue
        key=frozenset(chosen)
        if not any(key==frozenset(x) for x in found):found.append(np.array(chosen))
        if len(found)>=count:break
    return found


def pair_modes(edge,cfg,kind,seed):
    x,y,w=edge['x'],edge['y'],edge['weight'];N=len(x)
    if N<3:return []
    candidates=[]
    if kind=='ransac':
        rng=np.random.default_rng(seed)
        for _ in range(cfg['ransac_trials']):
            ix=rng.choice(N,3,replace=False,p=np.maximum(w,1e-8)/np.maximum(w,1e-8).sum())
            if np.max(abs(distances(x[ix])-distances(y[ix])))>cfg['geometry_tolerance_m']:continue
            T=rigid(x[ix],y[ix])
            if T is not None:candidates.append(T)
    else:
        for ix in cliques(compatible(x,y,cfg),w,cfg['modes']*2):
            T=rigid(x[ix],y[ix],w[ix])
            if T is not None:candidates.append(T)
    ranked=[]
    for T in candidates:
        for _ in range(3):
            residual=np.linalg.norm(apply(T,x)-y,axis=1);ix=np.flatnonzero(residual<cfg['inlier_m'])
            if len(ix)<3:break
            new=rigid(x[ix],y[ix],w[ix]/(1+(residual[ix]/cfg['inlier_m'])**2))
            if new is None:break
            T=new
        residual=np.linalg.norm(apply(T,x)-y,axis=1);value=float(np.sum(w*np.maximum(0,1-(residual/cfg['inlier_m'])**2)))
        ranked.append((value,T))
    ranked.sort(key=lambda a:-a[0]);result=[]
    for _,T in ranked:
        if any(np.median(np.linalg.norm(apply(T,x)-apply(U,x),axis=1))<.005 for U in result):continue
        result.append(T)
        if len(result)>=cfg['modes']:break
    return result


def material_tracks(edges,cfg):
    h2=defaultdict(dict);c12=defaultdict(dict)
    for q,r in enumerate(edges[1]):
        i,k=divmod(q,24)
        for h,(a,c) in enumerate(zip(r['source'],r['target'])):h2[int(a)][int(c)]=(i,k,r['x'][h],r['y'][h],float(r['weight'][h]))
    for q,r in enumerate(edges[2]):
        j,k=divmod(q,24)
        for h,(b,c) in enumerate(zip(r['source'],r['target'])):c12[int(b)][int(c)]=(j,k,r['x'][h],r['y'][h],float(r['weight'][h]))
    groups=defaultdict(list)
    for q,r in enumerate(edges[0]):
        i,j=divmod(q,24)
        for h,(a,b) in enumerate(zip(r['source'],r['target'])):
            aa,bb=h2.get(int(a),{}),c12.get(int(b),{})
            for c in aa.keys()&bb.keys():
                i2,k,x0,x2,w02=aa[c];j2,k2,x1,__,w12=bb[c]
                if i2!=i or j2!=j or k!=k2:continue
                assert np.linalg.norm(x0-r['x'][h])<1e-8 and np.linalg.norm(x1-r['y'][h])<1e-8
                groups[i,j,k].append((int(a),int(b),int(c),np.array([x0,x1,x2]),(w02*w12*float(r['weight'][h]))**(1/3)))
    result={};raw={}
    for key,rr in groups.items():
        raw[key]=len(rr);rr.sort(key=lambda x:-x[4]);rr=rr[:cfg['tracks_per_tuple']]
        x=np.array([a[3] for a in rr]);node=np.array([a[:3] for a in rr],np.int64);w=np.array([a[4] for a in rr])
        counts=[dict(zip(*np.unique(node[:,t],return_counts=True))) for t in range(3)]
        w/=np.array([max(counts[t][int(n[t])] for t in range(3)) for n in node])
        result[key]=dict(x=x,node=node,weight=w)
    return result,raw


def skew(x):
    out=np.zeros(x.shape[:-1]+(3,3));out[...,0,1]=-x[...,2];out[...,0,2]=x[...,1];out[...,1,0]=x[...,2];out[...,1,2]=-x[...,0];out[...,2,0]=-x[...,1];out[...,2,1]=x[...,0];return out


def residual_jacobian(T,x):
    G1,G2=T;x0,x1,x2=x[:,0],x[:,1],x[:,2];z1=apply(G1,x0);z2=apply(G2,x0);z12=apply(G2@inverse(G1),x1)
    r=np.array([z1-x1,z2-x2,z12-x2]);N=len(x);J=np.zeros((3,N,3,12));I=np.eye(3)
    J[0,:,:,:3]=-skew(z1);J[0,:,:,3:6]=I;J[1,:,:,6:9]=-skew(z2);J[1,:,:,9:12]=I
    R=G2[:3,:3]@G1[:3,:3].T;J[2,:,:,:3]=R@skew(x1);J[2,:,:,3:6]=-R;J[2,:,:,6:9]=-skew(z12);J[2,:,:,9:12]=I
    return r,J


def retract(T,delta):
    out=[]
    for k in range(2):
        D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(delta[k*6:k*6+3]).as_matrix();D[:3,3]=delta[k*6+3:k*6+6];out.append(D@T[k])
    return np.array(out)


def joint_fit(initial,x,weight,cfg):
    T=np.array(initial);w=weight/max(weight.sum(),1e-12);s=cfg['inlier_m'];path=[]
    def loss(T):
        r,_=residual_jacobian(T,x);return float(np.sum(w[None]*np.array([1,1,.5])[:,None]*s*s*np.log1p((r*r).sum(2)/(s*s))))
    current=loss(T);path.append(current)
    for _ in range(cfg['joint_steps']):
        r,J=residual_jacobian(T,x);factor=np.sqrt(w[None]*np.array([1,1,.5])[:,None]/(1+(r*r).sum(2)/(s*s)))
        A=(J*factor[:,:,None,None]).reshape(-1,12);b=(r*factor[:,:,None]).ravel()
        delta=np.linalg.lstsq(np.r_[A,np.eye(12)*.001],np.r_[-b,np.zeros(12)],rcond=None)[0];delta*=min(1,.05/max(np.linalg.norm(delta),1e-12));accepted=False
        for alpha in [1.,.5,.25,.125]:
            candidate=retract(T,delta*alpha);value=loss(candidate)
            if value<current-1e-14:T=candidate;current=value;accepted=True;break
        path.append(current)
        if not accepted:break
    return T,np.array(path)


def coupled_modes(track,pair_a,pair_b,cfg,kind):
    x,w=track['x'],track['weight'];N=len(x)
    if N<3:return []
    C01=compatible(x[:,0],x[:,1],cfg);C02=compatible(x[:,0],x[:,2],cfg);C12=compatible(x[:,1],x[:,2],cfg)
    # All track selectors obey the SAME three-epoch observation exclusivity.
    exclusive=(distances(x[:,0])<cfg['exclusion_m'])|(distances(x[:,1])<cfg['exclusion_m'])|(distances(x[:,2])<cfg['exclusion_m'])
    C01[exclusive]=0;C02[exclusive]=0;C12[exclusive]=0
    selections=[];initials=[]
    if kind=='joint_all':
        for a in pair_a[:2]:
            for b in pair_b[:2]:selections.append(np.arange(N));initials.append(np.array([a,b]))
        if not selections:selections=[np.arange(N)]
    elif kind=='cycle_independent':
        a=cliques(C01,w,cfg['modes']);b=cliques(C02,w,cfg['modes'])
        for k in range(max(len(a),len(b))):selections.append(np.unique(np.r_[a[k%len(a)] if a else [],b[k%len(b)] if b else []]).astype(int))
    else:selections=cliques(C01*C02*C12,w,cfg['modes'])
    result=[]
    for k,ix in enumerate(selections[:cfg['modes']]):
        if len(ix)<3:continue
        if k<len(initials):T=initials[k]
        else:
            a=rigid(x[ix,0],x[ix,1],w[ix]);b=rigid(x[ix,0],x[ix,2],w[ix])
            if a is None or b is None:continue
            T=np.array([a,b])
        initial=T.copy();T,path=joint_fit(T,x[ix],w[ix],cfg)
        if any(max(np.median(np.linalg.norm(apply(T[e],x[:,0])-apply(r['T'][e],x[:,0]),axis=1)) for e in [0,1])<.005 for r in result):continue
        result.append(dict(T=T,initial=initial,path=path,selected=ix))
    return result
