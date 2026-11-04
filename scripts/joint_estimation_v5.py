"""Point covariance fusion, variance-normalized proposal and association prototype.

Approximate: scalar covariance weights in SVD, not exact anisotropic SE3 optimization.
"""
import itertools
import numpy as np
from scripts.joint_estimation import assign,fit_rigid,transform

def skew(p):
    x,y,z=p;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def fuse_history(positions,covariances):
    precision=np.linalg.inv(covariances);cov=np.linalg.inv(precision.sum(axis=0))
    rhs=np.einsum('hnab,hnb->na',precision,positions)
    return np.einsum('nab,nb->na',cov,rhs),cov

def solve(obs,cfg,sampler,seed,diagnostics=False):
    ref,C=fuse_history(obs['history_positions'],obs['history_covariances']);features=obs['history_descriptors'].mean(axis=0);current=obs['current_positions'];desc=obs['current_descriptors']
    appearance=np.sum((desc[:,None]-features[None])**2,axis=2)/cfg['descriptor_scale']**2;A0=assign(appearance);rng=np.random.default_rng(seed)
    top=np.argsort(appearance,axis=1,kind='stable')[:,:min(sampler['top_k'],len(ref))];starts=[obs['initial_T']];a=np.array([0,0,1]);b=np.array([1,2,2]);sigma2=cfg['geometry_scale_m']**2
    current_variance=obs['current_variance_m2'];eps=sigma2
    for _ in range(cfg['proposal_count']):
        ids=rng.choice(len(current),3,replace=False);mapping=A0[ids];combos=np.array(list(itertools.product(*(top[i] for i in ids))))
        unique=(combos[:,0]!=combos[:,1])&(combos[:,0]!=combos[:,2])&(combos[:,1]!=combos[:,2]);combos=combos[unique];p=ref[combos]
        area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1);combos=combos[area>eps];p=ref[combos]
        source_area=np.linalg.norm(np.cross(current[ids[1]]-current[ids[0]],current[ids[2]]-current[ids[0]]))
        if len(combos) and source_area>eps:
            edges=p[:,a]-p[:,b];lengths=np.linalg.norm(edges,axis=2);directions=edges/np.maximum(lengths[...,None],1e-9)
            variance=np.einsum('nka,nkab,nkb->nk',directions,C[combos[:,a]]+C[combos[:,b]],directions)+2*current_variance+sigma2
            source_lengths=np.linalg.norm(current[ids[a]]-current[ids[b]],axis=1)
            edge_cost=np.sum((lengths-source_lengths)**2/variance+np.log(variance/sigma2),axis=1)
            descriptor=appearance[ids[None,:],combos].sum(axis=1);mapping=combos[np.argmin(edge_cost+cfg['descriptor_weight']*descriptor)]
        starts.append(fit_rigid(current[ids],ref[mapping]))
    Sigma=C+(sigma2+current_variance)*np.eye(3)[None];precision=np.linalg.inv(Sigma);logdet=np.linalg.slogdet(Sigma)[1]-3*np.log(sigma2);solutions=[]
    def costs(T):
        diff=transform(current,T)[:,None]-ref[None];q=np.einsum('mna,nab,mnb->mn',diff,precision,diff)
        return q+logdet[None]+cfg['descriptor_weight']*appearance,q
    for T in starts:
        T=T.copy()
        for _ in range(cfg['iterations']):
            cost,q=costs(T);A=assign(cost);standard=np.sqrt(np.maximum(q[np.arange(len(A)),A],0))*cfg['geometry_scale_m']
            switch=1/(1+(standard/cfg['switch_scale_m'])**4);weights=switch/np.trace(Sigma[A],axis1=1,axis2=2)
            T=fit_rigid(current,ref[A],weights)
        cost,q=costs(T);A=assign(cost);selected=q[np.arange(len(A)),A]
        score=float((np.minimum(selected,cfg['truncation_cost'])+logdet[A]+cfg['descriptor_weight']*appearance[np.arange(len(A)),A]).sum())
        r=np.linalg.norm(transform(current,T)-ref[A],axis=1);solutions.append(dict(T=T,assignment=A,residual_m=r,score=score))
    selected=min(range(len(solutions)),key=lambda i:solutions[i]['score']);s=solutions[selected]
    pred=dict(T=s['T'],assignment=s['assignment'],changed=s['residual_m']>cfg['change_threshold_m'],residual_m=s['residual_m'],hypotheses=len(starts))
    if diagnostics:pred.update(initial_poses=starts,solutions=solutions,selected_index=selected)
    return pred
