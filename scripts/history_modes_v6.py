"""Anchor-aware historical modes with current association; no truth inputs."""
import itertools
import numpy as np
from scripts.joint_estimation import assign,fit_rigid,transform
from scripts.joint_estimation_v4 import solve as solve_v4

def geomedian(points,iterations=30):
    x=np.mean(points,axis=0)
    for _ in range(iterations):
        distance=np.linalg.norm(points-x,axis=1);weights=1/np.maximum(distance,1e-6);x=(points*weights[:,None]).sum(axis=0)/weights.sum()
    return x

def clusters(points,radius):
    groups=[[i] for i in range(len(points))]
    while len(groups)>1:
        pairs=[(float(np.linalg.norm(points[a][:,None]-points[b][None],axis=2).max()),i,j) for i,a in enumerate(groups) for j,b in enumerate(groups) if i<j]
        distance,i,j=min(pairs)
        if distance>radius:break
        groups[i]=sorted(groups[i]+groups[j]);groups.pop(j)
    return groups

def memory(obs,method,mcfg):
    P=obs['history_positions'];F=obs['history_descriptors'];H,N,_=P.shape;anchor=P[0];anchor_features=F[0]
    if method=='anchor_only':return anchor,anchor_features
    if method=='geomedian':return np.array([geomedian(P[:,j]) for j in range(N)]),np.array([geomedian(F[:,j]) for j in range(N)])
    if method=='reassociate':
        positions=[];features=[]
        for p,f in zip(P,F):
            cost=((anchor[:,None]-p[None])**2).sum(axis=2)/.12**2+.25*((anchor_features[:,None]-f[None])**2).sum(axis=2)/.18**2
            permutation=assign(cost);positions.append(p[permutation]);features.append(f[permutation])
        return np.mean(positions,axis=0),np.mean(features,axis=0)
    if method=='anchor_cluster':
        positions=[];features=[]
        for j in range(N):
            groups=clusters(P[:,j],mcfg['cluster_radius_m']);group=next(g for g in groups if 0 in g)
            positions.append(P[group,j].mean(axis=0));features.append(F[group,j].mean(axis=0))
        return np.array(positions),np.array(features)
    return P.mean(axis=0),F.mean(axis=0)

def infer(obs,cfg,sampler,mcfg,method,seed):
    if method!='v6_modes':
        p,f=memory(obs,method,mcfg);a={**obs,'history_positions':p[None],'history_descriptors':f[None]};result=solve_v4(a,cfg,sampler,seed,'geometry',True)
        result['reference_points']=p[result['assignment']]
        return result
    P=obs['history_positions'];F=obs['history_descriptors'];N=P.shape[1];positions=[];features=[];prior=[];owners=[]
    for j in range(N):
        for group in clusters(P[:,j],mcfg['cluster_radius_m']):
            center=P[group,j].mean(axis=0);positions.append(center);features.append(F[group,j].mean(axis=0));owners.append(j)
            prior.append(-np.log(len(group)/len(P))+np.linalg.norm(center-P[0,j])**2/(2*mcfg['anchor_prior_sigma_m']**2))
    positions=np.array(positions);features=np.array(features);prior=np.array(prior);owners=np.array(owners);current=obs['current_positions'];desc=obs['current_descriptors']
    appearance=((desc[:,None]-features[None])**2).sum(axis=2)/cfg['descriptor_scale']**2
    fixed=cfg['descriptor_weight']*appearance+mcfg['prior_weight']*prior[None]
    # Candidate generation retains mode alternatives, restricted to distinct nominal IDs.
    rng=np.random.default_rng(seed);top=np.argsort(fixed,axis=1,kind='stable')[:,:min(sampler['top_k'],len(positions))];starts=[obs['initial_T']];a=np.array([0,0,1]);b=np.array([1,2,2]);sigma2=cfg['geometry_scale_m']**2
    fallback_points,fallback_features=memory(obs,'anchor_cluster',mcfg)
    fallback_A=assign(((desc[:,None]-fallback_features[None])**2).sum(axis=2)/cfg['descriptor_scale']**2)
    for _ in range(cfg['proposal_count']):
        ids=rng.choice(len(current),3,replace=False);combos=np.array(list(itertools.product(*(top[i] for i in ids))));o=owners[combos];combos=combos[(o[:,0]!=o[:,1])&(o[:,0]!=o[:,2])&(o[:,1]!=o[:,2])];p=positions[combos];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1);combos=combos[area>sigma2];p=positions[combos]
        source_area=np.linalg.norm(np.cross(current[ids[1]]-current[ids[0]],current[ids[2]]-current[ids[0]]))
        if len(combos) and source_area>sigma2:
            lengths=np.linalg.norm(p[:,a]-p[:,b],axis=2);s=np.linalg.norm(current[ids[a]]-current[ids[b]],axis=1);score=((lengths-s)**2/sigma2).sum(axis=1)+fixed[ids[None,:],combos].sum(axis=1);target=p[np.argmin(score)]
        else:target=fallback_points[fallback_A[ids]]
        starts.append(fit_rigid(current[ids],target))
    def costs(T):
        r2=((transform(current,T)[:,None]-positions[None])**2).sum(axis=2);cost=r2/sigma2+fixed
        reduced=np.empty((len(current),N));mode=np.empty((len(current),N),int)
        for j in range(N):
            modes=np.flatnonzero(owners==j);k=cost[:,modes].argmin(axis=1);mode[:,j]=modes[k];reduced[:,j]=cost[np.arange(len(current)),modes[k]]
        return reduced,mode
    solutions=[]
    for T in starts:
        T=T.copy()
        for _ in range(cfg['iterations']):
            cost,mode=costs(T);A=assign(cost);k=mode[np.arange(len(A)),A];reference=positions[k];r=np.linalg.norm(transform(current,T)-reference,axis=1);T=fit_rigid(current,reference,1/(1+(r/cfg['switch_scale_m'])**4))
        cost,mode=costs(T);A=assign(cost);k=mode[np.arange(len(A)),A];reference=positions[k];r=np.linalg.norm(transform(current,T)-reference,axis=1);score=float((np.minimum(r*r/sigma2,cfg['truncation_cost'])+fixed[np.arange(len(A)),k]).sum());solutions.append(dict(T=T,assignment=A,residual_m=r,score=score,reference_points=reference,mode=k))
    selected=min(range(len(solutions)),key=lambda i:solutions[i]['score']);s=solutions[selected]
    return dict(T=s['T'],assignment=s['assignment'],changed=s['residual_m']>cfg['change_threshold_m'],residual_m=s['residual_m'],reference_points=s['reference_points'],initial_poses=starts,solutions=solutions,selected_index=selected,hypotheses=len(starts),memory_modes=len(positions))
