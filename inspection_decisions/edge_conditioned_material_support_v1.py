"""Conditional finite axial edges, measured ray pairs, convex interval profiling.

Not a complete surface, calibrated interval, or original line-BA claim. The
persistent (signed plane, signed edge) key is an explicit working association.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES,right_jacobian


def plane_ray(frame,pixels,normal,centroid,center,cfg):
    pixels=np.asarray(pixels,float);h,w=frame['depth'].shape
    ray=np.c_[(pixels[:,0]-(w-1)/2)/cfg['focal_px'],(pixels[:,1]-(h-1)/2)/cfg['focal_px'],np.ones(len(pixels))]@frame['C'][:3,:3].T
    origin=frame['C'][:3,3]-center;den=ray@normal
    depth=np.divide((centroid-origin)@normal,den,out=np.full(len(ray),np.nan),where=np.abs(den)>cfg['edge_ray_transversality_min'])
    return origin+depth[:,None]*ray,depth,ray


def extract_edges(frame,lab,world,slot,center,planes,cfg,seed):
    """No grid hull. Background neighbor must actually return behind the plane."""
    yy,xx=np.where(lab==slot);rng=np.random.default_rng(seed);records=[];log={'tested_neighbors':0,'occlusion_or_unresolved_censored':0,'farther_ray_witnesses':0,'segments':0}
    h,w=lab.shape
    for f in planes:
        if f['view']!=frame['edge_view']:continue
        members=np.asarray(f['measured_indices'],int);pixels=np.c_[xx[members],yy[members]];pairs=[]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            other=pixels+[dx,dy];inside=(other[:,0]>=0)&(other[:,0]<w)&(other[:,1]>=0)&(other[:,1]<h)
            p=pixels[inside];q=other[inside];take=lab[q[:,1],q[:,0]]!=slot;p=p[take];q=q[take];log['tested_neighbors']+=len(p)
            if not len(p):continue
            qi,di,_=plane_ray(frame,p,f['normal'],f['centroid'],center,cfg);qo,do,_=plane_ray(frame,q,f['normal'],f['centroid'],center,cfg)
            measured=frame['depth'][q[:,1],q[:,0]]
            valid=np.isfinite(di)&np.isfinite(do)&(di>0)&(do>0)&(measured-do>cfg['edge_farther_return_margin_m'])
            # Positive plane member must also agree with the fitted ray return.
            valid&=np.abs(di-frame['depth'][p[:,1],p[:,0]])<cfg['edge_positive_fit_margin_m']
            log['occlusion_or_unresolved_censored']+=int((~valid).sum());log['farther_ray_witnesses']+=int(valid.sum())
            for pp,qq,a,b in zip(p[valid],q[valid],qi[valid],qo[valid]):pairs.append((pp,qq,a,b))
        if len(pairs)<cfg['edge_min_witnesses']:continue
        pixels=np.array([(.5*(p[0]+p[1])) for p in pairs]);remaining=np.arange(len(pairs))
        while len(remaining)>=cfg['edge_min_witnesses']:
            best=None
            for _ in range(cfg['edge_line_ransac_iterations']):
                a,b=rng.choice(remaining,2,replace=False);d=pixels[b]-pixels[a];length=np.linalg.norm(d)
                if length<cfg['edge_min_span_px']:continue
                tangent=d/length;delta=pixels[remaining]-pixels[a];distance=np.abs(tangent[0]*delta[:,1]-tangent[1]*delta[:,0]);ids=remaining[distance<=cfg['edge_line_tolerance_px']]
                if len(ids)<cfg['edge_min_witnesses']:continue
                span=np.ptp(pixels[ids]@tangent)
                if span<cfg['edge_min_span_px']:continue
                if best is None or len(ids)>len(best):best=ids
            if best is None:break
            p2=pixels[best];mean=p2.mean(0);_,_,V=np.linalg.svd(p2-mean,full_matrices=False);line_normal=V[-1];line=np.r_[line_normal,-line_normal@mean]
            camera_plane=frame['C'][:3,:3]@np.array([line[0]*cfg['focal_px'],line[1]*cfg['focal_px'],line[2]+line[0]*(w-1)/2+line[1]*(h-1)/2])
            tangent3=np.cross(f['normal'],camera_plane);tangent3/=max(np.linalg.norm(tangent3),1e-15)
            edge_normal=np.cross(tangent3,f['normal']);edge_normal/=max(np.linalg.norm(edge_normal),1e-15)
            qi=np.array([pairs[j][2] for j in best]);qo=np.array([pairs[j][3] for j in best]);step=(qo-qi).mean(0)
            if edge_normal@step<0:edge_normal=-edge_normal
            k=int(np.argmax(AXES@edge_normal));aligned=AXES[k]@edge_normal
            # Disconnected collinear pieces remain separate local segments.
            order=best[np.argsort(pixels[best]@V[0])];s=pixels[order]@V[0];groups=np.split(order,np.flatnonzero(np.diff(s)>cfg['edge_segment_gap_px'])+1)
            for group in groups:
                if len(group)<cfg['edge_min_witnesses'] or np.ptp(pixels[group]@V[0])<cfg['edge_min_span_px']:continue
                if k%3==f['axis']%3 or aligned<cfg['edge_axis_cosine_min']:continue
                # Keep every witness; weights normalize by segment size.
                pi=np.array([pairs[j][0] for j in group]);po=np.array([pairs[j][1] for j in group]);a=np.array([pairs[j][2] for j in group]);b=np.array([pairs[j][3] for j in group])
                if np.any((b-a)@AXES[k]<=0):continue
                records.append({'key':(int(f['axis']),k),'view':f['view'],'inside':a,'outside':b,'inside_pixels':pi,'outside_pixels':po,'plane_normal':f['normal'],'plane_centroid':f['centroid'],'axis_cosine':float(aligned),'line_pixels':line,'witnesses':len(group)})
            remaining=remaining[~np.isin(remaining,best)]
    log['segments']=len(records);return records,log


def endpoints(x,features,cfg,jacobian=False):
    R=Rotation.from_rotvec(x[:3]).as_matrix();Jr=right_jacobian(x[:3]);result={}
    for f in features:
        key=tuple(f['key']);n=AXES[key[1]];N=len(f['inside']);v=np.concatenate([f['inside'],f['outside']])-x[3:];q=v@(R@n)
        low=q[:N]-cfg['edge_interval_margin_m'];high=q[N:]+cfg['edge_interval_margin_m'];weight=np.full(N,1./N)
        J=None
        if jacobian:
            J=np.zeros((2*N,6));J[:,:3]=np.cross(n,v@R)@Jr;J[:,3:]=-(R@n)
        result.setdefault(key,[]).append((low,high,weight,J[:N] if J is not None else None,J[N:] if J is not None else None))
    return {k:tuple(np.concatenate([v[i] for v in vv]) if i<3 or jacobian else None for i in range(5)) for k,vv in result.items()}


def profile_intervals(low,high,weight,jlow=None,jhigh=None):
    """Vectorized convex profile along last dimension, including inverted bounds."""
    low=np.asarray(low);high=np.asarray(high);weight=np.broadcast_to(weight,low.shape)
    a=np.minimum(low,high).min(-1);b=np.maximum(low,high).max(-1)
    for _ in range(44):
        mid=(a+b)/2;g=np.sum(weight*(np.minimum(mid[...,None]-low,0)+np.maximum(mid[...,None]-high,0)),axis=-1)
        a=np.where(g<0,mid,a);b=np.where(g>0,mid,b)
        # When g==0 the entire local intersection is optimal; freeze it.
        a=np.where(g==0,mid,a);b=np.where(g==0,mid,b)
    value=(a+b)/2;active_low=low>value[...,None];active_high=high<value[...,None]
    rlow=np.maximum(low-value[...,None],0);rhigh=np.maximum(value[...,None]-high,0)
    cost=np.sum(weight*(rlow**2+rhigh**2),axis=-1)
    if jlow is None:return cost,value
    assert low.ndim==1
    active_weight=weight*(active_low.astype(float)+active_high.astype(float));den=active_weight.sum()
    Jb=np.sum(weight[:,None]*(active_low[:,None]*jlow+active_high[:,None]*jhigh),axis=0)/den if den>0 else np.zeros(jlow.shape[-1])
    residual=np.r_[np.sqrt(weight)*rlow,np.sqrt(weight)*rhigh]
    J=np.r_[np.sqrt(weight)[:,None]*active_low[:,None]*(jlow-Jb),np.sqrt(weight)[:,None]*active_high[:,None]*(Jb-jhigh)]
    return residual,J,float(value)


def edge_residual(parameters,features,history,cfg,shared=True):
    current=[endpoints(parameters[6*t:6*t+6],ff,cfg,True) for t,ff in enumerate(features)];historical=endpoints(np.zeros(6),history,cfg,True)
    rows=[];jac=[]
    keys=set(historical)|set(current[0])|set(current[1])
    for key in sorted(keys):
        for periods in ([[0,1]] if shared else [[0],[1]]):
            low=[];high=[];weights=[];jl=[];jh=[]
            for t in periods:
                for obj,fixed in [(historical,True),(current[t],False)]:
                    if key not in obj:continue
                    l,h,w,L,H=obj[key];low.extend(l);high.extend(h);weights.extend(w);A=np.zeros((len(l),12));B=A.copy()
                    if not fixed:A[:,6*t:6*t+6]=L;B[:,6*t:6*t+6]=H
                    jl.extend(A);jh.extend(B)
            if not low:continue
            r,J,_=profile_intervals(np.array(low),np.array(high),np.array(weights),np.array(jl),np.array(jh));rows.extend(r/cfg['sigma_edge_m']);jac.extend(J/cfg['sigma_edge_m'])
    return np.array(rows),np.array(jac).reshape(-1,12)


def edge_pair_costs(candidates,features,centers,history,cfg):
    """Full candidate matrices; no candidate/identity gate and no GT."""
    from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
    current=[[endpoints(pose_vector(b['T'],centers[b['slot']]),features[t][b['slot']],cfg) for b in bb] for t,bb in enumerate(candidates)]
    hist=endpoints(np.zeros(6),history,cfg);N,M=map(len,candidates);independent=np.zeros((N,M));shared=np.zeros((N,M))
    keys=set(hist)|set().union(*(set(s) for vv in current for s in vv))
    for key in sorted(keys):
        arrays=[]
        # Variable support by slot is packed with zero-weight padding.
        for vv in current:
            size=max((len(s[key][0]) if key in s else 0) for s in vv);l=np.zeros((len(vv),size));h=l.copy();w=l.copy()
            for i,s in enumerate(vv):
                if key in s:
                    a,b,c,_,_=s[key];l[i,:len(a)]=a;h[i,:len(a)]=b;w[i,:len(a)]=c
            arrays.append((l,h,w))
        hl,hh,hw=(hist[key][:3] if key in hist else (np.empty(0),np.empty(0),np.empty(0)))
        unary=[]
        for l,h,w in arrays:
            ll=np.c_[np.broadcast_to(hl,(len(l),len(hl))),l];hi=np.c_[np.broadcast_to(hh,(len(h),len(hh))),h];ww=np.c_[np.broadcast_to(hw,(len(w),len(hw))),w]
            unary.append(profile_intervals(ll,hi,ww)[0] if ll.shape[1] else np.zeros(len(l)))
        independent+=unary[0][:,None]+unary[1][None,:]
        # Flatten full matrix into chunks to bound memory, preserving all pairs.
        target=shared.ravel();ii,jj=np.indices((N,M));ii=ii.ravel();jj=jj.ravel()
        for start in range(0,N*M,cfg['edge_profile_chunk_size']):
            I=ii[start:start+cfg['edge_profile_chunk_size']];K=jj[start:start+cfg['edge_profile_chunk_size']];size=len(I)
            lows=[np.broadcast_to(hl,(size,len(hl)))]*2+[arrays[0][0][I],arrays[1][0][K]]
            highs=[np.broadcast_to(hh,(size,len(hh)))]*2+[arrays[0][1][I],arrays[1][1][K]]
            weights=[np.broadcast_to(hw,(size,len(hw)))]*2+[arrays[0][2][I],arrays[1][2][K]]
            lo=np.concatenate(lows,axis=1)
            if lo.shape[1]:target[start:start+size]+=profile_intervals(lo,np.concatenate(highs,axis=1),np.concatenate(weights,axis=1))[0]
    return independent/cfg['sigma_edge_m']**2,shared/cfg['sigma_edge_m']**2
