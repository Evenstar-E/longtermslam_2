"""Conventional historical frame realignment using frozen v4; a strong control.

Reference construction sees historical data only; current measurements never select
historical modes. This reuses established registration, not a novelty claim.
"""
import numpy as np
from scripts.joint_estimation import transform
from scripts.joint_estimation_v4 import solve

def infer(obs,cfg,sampler,seed):
    P=obs['history_positions'];F=obs['history_descriptors'];positions=[P[0]];features=[F[0]]
    for h in range(1,len(P)):
        historical=dict(history_positions=P[:1],history_descriptors=F[:1],current_positions=P[h],current_descriptors=F[h],initial_T=np.eye(4))
        fitted=solve(historical,cfg,sampler,seed+10000*h,'geometry');p=transform(P[h],fitted['T']);ordered=np.empty_like(p);ordered[fitted['assignment']]=p;f=np.empty_like(F[h]);f[fitted['assignment']]=F[h];positions.append(ordered);features.append(f)
    reference=np.mean(positions,axis=0);descriptor=np.mean(features,axis=0)
    repaired={**obs,'history_positions':reference[None],'history_descriptors':descriptor[None]};pred=solve(repaired,cfg,sampler,seed,'geometry',True);pred['reference_points']=reference[pred['assignment']]
    return pred
