"""Translation-only association with an explicit shared local-motion objective.

No file access or evaluator dependency. This is a known MAP-style model test,
not a claim that sparse penalties or block coordinate descent are novel.
"""
import numpy as np
from inspection_decisions.correspondence_joint_v1 import assign, report


def profile_cost(d, sigma, activation, tau):
    r2=np.sum(np.asarray(d)**2,axis=-1)
    return np.minimum(r2/sigma**2,activation+r2/(sigma**2+tau**2))


def frames_from(data):
    n=len(data['references'])
    return [(np.asarray(f['D']).reshape(-1,n,3),np.asarray(f['S']).reshape(-1,n),
             np.asarray(f['valid'],bool).reshape(-1,n)) for f in data['frames']]


def motion_fit(offsets,t,cfg,no_amplitude=False):
    if not len(offsets):return np.zeros(3)
    a=np.asarray(offsets);k=len(a);d=a.mean(0)-t;s2=cfg['sigma']**2
    shrink=1. if no_amplitude else k*cfg['motion_scale_m']**2/(s2+k*cfg['motion_scale_m']**2)
    moving=shrink*d
    gain=k*np.dot(d,d)/s2
    moving_cost=k*np.dot(d-moving,d-moving)/s2
    if not no_amplitude:moving_cost+=np.dot(moving,moving)/cfg['motion_scale_m']**2
    return moving if gain-moving_cost>cfg['change_penalty'] else np.zeros(3)


def energy(frames,matches,t,delta,cfg,no_amplitude=False):
    value=cfg['change_penalty']*sum(np.dot(d,d)>1e-20 for d in delta)
    if not no_amplitude:value+=np.sum(delta**2)/cfg['motion_scale_m']**2
    for (D,S,V),pairs in zip(frames,matches):
        value+=12*(len(D)-len(pairs))
        for i,j in pairs:value+=np.sum((D[i,j]-t-delta[j])**2)/cfg['sigma']**2+S[i,j]/.1
    return float(value)


def seeds_from(frames,cfg):
    samples=np.concatenate([D[V] for D,S,V in frames]);grid=np.round(samples/.1).astype(int)
    cells,count=np.unique(grid,axis=0,return_counts=True)
    seeds=[np.zeros(3)]+[c*.1 for c in cells[np.argsort(-count,kind='stable')[:cfg['max_seeds']]]]
    return [s for s in seeds if np.max(abs(s))<=cfg['public_session_translation_bound_m']]


def fit_shared(frames,initial,cfg,no_amplitude=False):
    n=frames[0][0].shape[1];t=np.array(initial);delta=np.zeros((n,3));trace=[]
    F=max(len(frames),1);tau=np.inf if no_amplitude else cfg['motion_scale_m']*np.sqrt(F)
    # Same independently profiled initialization as the ordinary regularized control.
    matches=[assign(profile_cost(D-t,cfg['sigma'],cfg['change_penalty']/F,tau)+S/.1,V) for D,S,V in frames]
    trace.append(energy(frames,matches,t,delta,cfg,no_amplitude))
    for _ in range(cfg['iterations']):
        per=[[] for _ in range(n)]
        for (D,S,V),pairs in zip(frames,matches):
            for i,j in pairs:per[j].append(D[i,j])
        for j,v in enumerate(per):delta[j]=motion_fit(v,t,cfg,no_amplitude)
        trace.append(energy(frames,matches,t,delta,cfg,no_amplitude))
        offsets=[D[i,j]-delta[j] for (D,S,V),pairs in zip(frames,matches) for i,j in pairs]
        if offsets:t=np.clip(np.mean(offsets,0),-cfg['public_session_translation_bound_m'],cfg['public_session_translation_bound_m'])
        trace.append(energy(frames,matches,t,delta,cfg,no_amplitude))
        matches=[assign(np.sum((D-t-delta[None])**2,axis=2)/cfg['sigma']**2+S/.1,V) for D,S,V in frames]
        trace.append(energy(frames,matches,t,delta,cfg,no_amplitude))
    # Each block is scored by the same objective, including local-motion cost.
    assert np.max(np.diff(trace),initial=0)<1e-7
    return dict(translation=t.tolist(),delta=delta.tolist(),matches=matches,score=trace[-1],energy_trace=trace)


def fit_profile(frames,initial,cfg):
    """Ordinary independent pair-motion profiling; same prior per full support."""
    t=np.array(initial);n=frames[0][0].shape[1];F=max(len(frames),1)
    activation=cfg['change_penalty']/F;tau=cfg['motion_scale_m']*np.sqrt(F);s2=cfg['sigma']**2
    trace=[]
    for _ in range(cfg['iterations']):
        matches=[assign(profile_cost(D-t,cfg['sigma'],activation,tau)+S/.1,V) for D,S,V in frames]
        fitted=[];per=[[] for _ in range(n)]
        for (D,S,V),pairs in zip(frames,matches):
            for i,j in pairs:
                d=D[i,j]-t
                moving=activation+np.dot(d,d)/(s2+tau**2)<np.dot(d,d)/s2
                z=tau**2/(s2+tau**2)*d if moving else np.zeros(3)
                fitted.append(D[i,j]-z);per[j].append(z)
        if fitted:t=np.clip(np.mean(fitted,0),-cfg['public_session_translation_bound_m'],cfg['public_session_translation_bound_m'])
    # Re-profile at the actual final t; independent costs and reported shifts agree.
    matches=[assign(profile_cost(D-t,cfg['sigma'],activation,tau)+S/.1,V) for D,S,V in frames]
    score=0.;per=[[] for _ in range(n)]
    for (D,S,V),pairs in zip(frames,matches):
        score+=12*(len(D)-len(pairs))
        for i,j in pairs:
            d=D[i,j]-t;score+=float(profile_cost(d,cfg['sigma'],activation,tau))+S[i,j]/.1
            moving=activation+np.dot(d,d)/(s2+tau**2)<np.dot(d,d)/s2
            per[j].append(tau**2/(s2+tau**2)*d if moving else np.zeros(3))
    return dict(translation=t.tolist(),delta=[np.median(v,0).tolist() if v else [0.,0.,0.] for v in per],matches=matches,score=float(score))


def solve(data,method,cfg):
    frames=frames_from(data);seeds=seeds_from(frames,cfg)
    fn=fit_profile if method=='independent_profile' else fit_shared
    solutions=[fn(frames,s,cfg) if method=='independent_profile' else fn(frames,s,cfg,method=='shared_no_amplitude') for s in seeds]
    unique={}
    for s in sorted(solutions,key=lambda s:s['score']):
        signature=tuple(tuple(tuple(p) for p in pairs) for pairs in s['matches']);unique.setdefault(signature,s)
    ranked=sorted(unique.values(),key=lambda s:s['score']);best=ranked[0]
    # Apply identical branch-ambiguity reporting to both new candidate and control.
    branches=[s for s in ranked if s['score']<=best['score']+cfg['branch_tolerance']]
    return dict(prediction=report(data,best,branches),best=best,branches=branches,
                distinct_assignment_hypotheses=len(ranked),seed_count=len(seeds))
