"""Full geometry assignment; robust pose and independent robust hypothesis ranking.

Same scales and search budget as v2. Optional variants support causal ablations.
"""
import numpy as np
from scripts.joint_estimation import fit_rigid,transform,assign

def solve(obs,cfg,seed,variant='v3',diagnostics=False):
    no_history=variant=='no_history';positions=obs['history_positions'][:1] if no_history else obs['history_positions'];features=obs['history_descriptors'][:1] if no_history else obs['history_descriptors']
    ref=positions.mean(axis=0);features=features.mean(axis=0);current=obs['current_positions'];desc=obs['current_descriptors']
    appearance=np.sum((desc[:,None]-features[None])**2,axis=2)/cfg['descriptor_scale']**2
    A0=assign(appearance);rng=np.random.default_rng(seed);starts=[obs['initial_T']]
    if variant!='single':
        for _ in range(cfg['proposal_count']):
            ids=rng.choice(len(current),3,replace=False);starts.append(fit_rigid(current[ids],ref[A0[ids]]))
    def cost(T):return cfg['descriptor_weight']*appearance+(np.linalg.norm(transform(current,T)[:,None]-ref[None],axis=2)/cfg['geometry_scale_m'])**2
    solutions=[]
    for start in starts:
        T=start.copy()
        for _ in range(cfg['iterations']):
            A=assign(cost(T));r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
            w=np.ones(len(r)) if variant=='no_weights' else 1/(1+(r/cfg['switch_scale_m'])**4)
            T=fit_rigid(current,ref[A],w)
        A=assign(cost(T));r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
        full=float(cost(T)[np.arange(len(A)),A].sum())
        robust=float((cfg['descriptor_weight']*appearance[np.arange(len(A)),A]+np.minimum((r/cfg['geometry_scale_m'])**2,cfg['truncation_cost'])).sum())
        solutions.append(dict(T=T,assignment=A,residual_m=r,full_score=full,robust_score=robust))
    key='full_score' if variant in ['v2','no_weights'] else 'robust_score'
    best=min(range(len(solutions)),key=lambda i:solutions[i][key]);s=solutions[best]
    pred=dict(T=s['T'],assignment=s['assignment'],changed=s['residual_m']>cfg['change_threshold_m'],residual_m=s['residual_m'],hypotheses=len(solutions))
    if diagnostics:pred['solutions']=solutions;pred['selected_index']=best
    return pred
