"""Common observation-only fragment adapter for excess connected components.

Engineering interface repair, not part of the proposed estimator contribution.
"""
import numpy as np
from scripts.joint_estimation import transform


def merge_excess_fragments(obs,points_per_object):
    if len(obs['query'])<=len(obs['bounds']):return obs,[[i] for i in range(len(obs['query']))]
    groups={}
    for i,q in enumerate(obs['query']):
        appearance=np.sum((obs['reference_features']-q['feature'])**2,axis=1)
        candidates=np.flatnonzero(appearance<=appearance.min()+1e-8)
        centre=transform(q['points'],obs['initial_T']).mean(axis=0)
        j=int(candidates[np.argmin(np.linalg.norm(obs['bounds'][candidates].mean(axis=1)-centre,axis=1))])
        groups.setdefault(j,[]).append(i)
    out=dict(obs);query=[];ownership=[]
    for members in groups.values():
        pix=np.unique(np.concatenate([obs['query'][i]['all_pixels'] for i in members]));p=obs['rays'][pix]*obs['depth'].ravel()[pix,None]
        take=np.linspace(0,len(p)-1,min(points_per_object,len(p)),dtype=int)
        query.append(dict(points=p[take],all_pixels=pix,feature=obs['query'][members[0]]['feature']))
        ownership.append(members)
    out['query']=query
    return out,ownership
