"""Decouple correspondence ranking from changed-anchor pose suppression."""
import numpy as np
from scripts.joint_estimation import fit_rigid,transform,assign

def infer_decoupled(obs,cfg,seed):
    ref=obs['history_positions'].mean(axis=0);features=obs['history_descriptors'].mean(axis=0)
    current=obs['current_positions'];desc=obs['current_descriptors']
    appearance=np.sum((desc[:,None]-features[None])**2,axis=2)/cfg['descriptor_scale']**2
    initial_assignment=assign(appearance);rng=np.random.default_rng(seed);proposals=[obs['initial_T']]
    for _ in range(cfg['proposal_count']):
        ids=rng.choice(len(current),3,replace=False);proposals.append(fit_rigid(current[ids],ref[initial_assignment[ids]]))
    def cost(T):
        residual=np.linalg.norm(transform(current,T)[:,None]-ref[None],axis=2)
        return cfg['descriptor_weight']*appearance+(residual/cfg['geometry_scale_m'])**2
    solutions=[]
    for T in proposals:
        T=T.copy()
        for _ in range(cfg['iterations']):
            A=assign(cost(T));r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
            T=fit_rigid(current,ref[A],1/(1+(r/cfg['switch_scale_m'])**4))
        A=assign(cost(T));solutions.append((float(cost(T)[np.arange(len(A)),A].sum()),T,A))
    _,T,A=min(solutions,key=lambda x:x[0]);r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
    return dict(T=T,assignment=A,changed=r>cfg['change_threshold_m'],residual_m=r,hypotheses=len(proposals))
