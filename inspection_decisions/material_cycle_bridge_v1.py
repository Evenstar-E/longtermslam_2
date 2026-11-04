"""Partial material-path witnesses, distinct from SE(3) path consistency.

Fixed bridge estimates are conditioning variables, not independent truth.
All distances and capacities concern discrete patches, not physical area.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from inspection_decisions.real_temporal_v1 import moved, fit_stats
from inspection_decisions.shared_contamination_transport_v1 import patches
from inspection_decisions.ray_material_evidence_v1 import ray_states


def partial_map(a, b, T, cfg):
    x, ca, va = a; y, cb, vb = b
    variance = cfg['sensor_sigma_m']**2 + va[:, None] + vb[None, :]
    cost = cdist(moved(x, T), y, 'sqeuclidean') / variance
    cost += cdist(ca, cb, 'sqeuclidean') / cfg['color_sigma']**2
    reward = np.maximum(cfg['match_squared_radius'] - cost, 0)
    ii, jj = linear_sum_assignment(-reward)
    good = reward[ii, jj] > 0
    result = np.full(len(x), -1, int); result[ii[good]] = jj[good]
    return result


def compose_maps(ab, bc):
    result = np.full(len(ab), -1, int)
    valid = ab >= 0; result[valid] = bc[ab[valid]]
    return result


def material_losses(direct, path, target, states, cfg):
    """A composed track is evidence; an absent track is not a disagreement.

    Relax exact index equality only by the declared patch discretization radius.
    Unknown-ray censoring acts ONLY on missing direct matches, never an observed
    correspondence mismatch. This prevents hiding an explicit mismatch as unknown.
    """
    y, _, var = target; eligible = path >= 0
    both = eligible & (direct >= 0)
    agrees = np.zeros(len(path), bool)
    ids = np.flatnonzero(both)
    if len(ids):
        dist2 = np.sum((y[direct[ids]]-y[path[ids]])**2, axis=1)
        tolerance2 = cfg['sensor_sigma_m']**2 + var[direct[ids]]+var[path[ids]]
        agrees[ids] = dist2 <= tolerance2
    raw = np.where(eligible, np.where(agrees, -1., 1.), 0.)
    censored = raw.copy()
    missing_unknown = eligible & (direct < 0) & (states == 0)
    censored[missing_unknown] = 0.
    return float(raw.mean()), float(censored.mean()), dict(
        eligible=int(eligible.sum()), agreed=int(agrees.sum()),
        matched_mismatch=int((both & ~agrees).sum()),
        absent_unknown=int(missing_unknown.sum()),
        absent_known=int((eligible & (direct < 0) & (states != 0)).sum()))


class BridgeModel:
    def __init__(self, history, current, bridges, cfg):
        self.h=history; self.c=current; self.cfg=cfg
        self.hp=patches(history,cfg['points']); self.cp=patches(current,cfg['points'])
        self.bridges=[]
        for frame, hs, sc in bridges:
            sp=patches(frame,cfg['points'])
            hsmap=partial_map(self.hp,sp,hs,cfg); scmap=partial_map(sp,self.cp,sc,cfg)
            q=compose_maps(hsmap,scmap)
            self.bridges.append(dict(frame=frame,patch=sp,hs=hs,sc=sc,path=sc@hs,q=q,
                hs_matches=int((hsmap>=0).sum()),sc_matches=int((scmap>=0).sum())))

    def score(self,T,base):
        cfg=self.cfg; direct=partial_map(self.hp,self.cp,T,cfg)
        tx=moved(self.hp[0],T)
        states=ray_states(tx,self.c.depth,self.c.mask,self.c.K,cfg['sensor_sigma_m'],[.001,5])[0]
        pose=[]; geom=[]; raw=[]; censored=[]; diag=[]
        for b in self.bridges:
            err=np.sum((tx-moved(self.hp[0],b['path']))**2,axis=1)
            pose.append(float(np.mean(np.minimum(err/cfg['sensor_sigma_m']**2,1))))
            geom.append(fit_stats(b['patch'][0],self.cp[0],T@np.linalg.inv(b['hs']),cfg['sensor_sigma_m'])['cost'])
            r,c,d=material_losses(direct,b['q'],self.cp,states,cfg)
            raw.append(r); censored.append(c); diag.append(d)
        w=cfg['cycle_weight']
        scores=dict(pair_only=float(base),multiview_nn=float(base+w*np.mean(geom)),
            pose_cycle=float(base+w*np.mean(pose)),material_cycle=float(base+w*np.mean(raw)),
            visible_material_cycle=float(base+w*np.mean(censored)))
        return scores, dict(bridges=diag,direct_matches=int((direct>=0).sum()),
            geometry_path_loss=float(np.mean(geom)),pose_path_loss=float(np.mean(pose)),
            material_loss=float(np.mean(raw)),visible_material_loss=float(np.mean(censored)))
