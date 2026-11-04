"""Restricted model-input adapter: acquisition groups and private labels stay out.

The generated close-up groups deliberately supervise material correspondence.
They are not an input identity tracker. Models receive only these tensor fields.
"""
import json
from pathlib import Path
import numpy as np


RGB_KEYS={'flat':'rgb_flat','rich':'rgb','repeated':'rgb_repeated','changed':'rgb_changed'}


def load_observation(path, variant='rich'):
    with np.load(path,allow_pickle=False) as f:
        # The raw caster's 30m no-hit sentinel is not a measured background plane.
        valid=(f['depth']>0)&(f['depth']<29.99)
        return dict(rgb=f[RGB_KEYS[variant]],depth=np.where(valid,f['depth'],0.),depth_valid=valid,K=f['K'],C=f['C'],
                    anonymous_slot=f['anonymous_slot'])


def load_scene_epoch(root,scene_name,epoch,variant='rich',shuffle_seed=0):
    dest=Path(root)/scene_name
    records=json.loads((dest/'frames.json').read_text(encoding='utf-8'))
    records=[r for r in records if r['epoch']==epoch]
    rng=np.random.default_rng(shuffle_seed)
    order=rng.permutation(len(records))
    # frame indices are returned separately for evaluator bookkeeping, never features.
    return [load_observation(dest/records[i]['observation'],variant) for i in order], [records[i]['index'] for i in order]


def load_training_pair(root,scene_name,pair_index,variant='rich'):
    dest=Path(root)/scene_name
    meta=json.loads((dest/'public_metadata.json').read_text(encoding='utf-8'))
    if meta['split']!='train':
        raise ValueError('Training adapter refuses validation/test supervision.')
    pairs=json.loads((dest/'supervision/pairs_v3.json').read_text(encoding='utf-8'))
    p=pairs[pair_index]
    a=load_observation(dest/'observations'/f'frame_{p["source"]:03d}.npz',variant)
    b=load_observation(dest/'observations'/f'frame_{p["target"]:03d}.npz',variant)
    with np.load(dest/p['labels'],allow_pickle=False) as f:labels=dict(f)
    # Labels include the physically same point, even when the changed variant repaints it.
    # appearance_changed_in_changed_variant supervises paint persistence separately.
    return {'source':a,'target':b}, labels
