"""Source-exclusive surface updates with reversible ownership and rooted support.

Deterministic alternating surrogate, NOT a calibrated likelihood or convergent EM.
Each target epoch queries only immutable history and the OTHER epoch. Connectivity
is a testable modeling assumption, not a proof of physical identity.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def transform(x, T):
    return x @ T[:3, :3].T + T[:3, 3]


def inverse(x, T):
    return (x - T[:3, 3]) @ T[:3, :3]


def voxelize(x, rgb, cls, size):
    _, first, inv, count = np.unique(np.floor(x / size).astype(np.int64),
                                    axis=0, return_index=True, return_inverse=True, return_counts=True)
    # Representative actual observations, not fabricated voxel centroids.
    return {'xyz':x[first].astype(float), 'rgb':rgb[first].astype(float)/255,
            'class':cls[first], 'raw_index':first, 'inverse':inv, 'mass':count.astype(float)}


def fit_rigid(current, canonical, weights):
    weights = weights / weights.sum()
    a = np.sum(current * weights[:,None], axis=0)
    b = np.sum(canonical * weights[:,None], axis=0)
    u, _, vt = np.linalg.svd(((current-a)*weights[:,None]).T @ (canonical-b))
    d = np.eye(3); d[-1,-1] = np.linalg.det(u @ vt)
    # current -> canonical, then invert for canonical -> current convention.
    r = vt.T @ d @ u.T
    invT = np.eye(4); invT[:3,:3] = r; invT[:3,3] = b-r@a
    return np.linalg.inv(invT)


def root_mask(x, rgb, history, cfg):
    """Components of observed SAME-source points must touch immutable history.

    An unsupported cross-period pair cannot certify itself. Edges and roots use
    measured geometry/color only. This intentionally forbids disconnected novel
    views without a within-source path; that may reject beneficial completion.
    """
    if not len(x): return np.zeros(0, bool), {'components':0, 'rooted_components':0}
    d, k = history['tree'].query(x, workers=1)
    seed = (d <= cfg['distance_scale_m']) & (np.linalg.norm(rgb-history['rgb'][k],axis=1) <= cfg['graph_rgb_max'])
    pairs = cKDTree(x).query_pairs(cfg['graph_radius_m'], output_type='ndarray')
    if len(pairs):
        pairs = pairs[np.linalg.norm(rgb[pairs[:,0]]-rgb[pairs[:,1]],axis=1) <= cfg['graph_rgb_max']]
    if len(pairs):
        rows=np.r_[pairs[:,0],pairs[:,1]]; cols=np.r_[pairs[:,1],pairs[:,0]]
        graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(x),len(x))).tocsr()
        n, cc=connected_components(graph,directed=False)
    else: n,cc=len(x),np.arange(len(x))
    counts=np.bincount(cc,weights=seed.astype(int),minlength=n)
    good=counts>=cfg['minimum_root_voxels']
    return good[cc], {'components':int(n),'rooted_components':int(good.sum()),'root_voxels':int(seed.sum())}


def make_surface(obs, labels, poses, history, cfg, rooted=False):
    increments=[]; stats=[]
    for i,h in enumerate(history):
        idx=np.flatnonzero(labels==i); x=inverse(obs['xyz'][idx], poses[i]); rgb=obs['rgb'][idx]
        keep,diagnostic=root_mask(x,rgb,h,cfg) if rooted else (np.ones(len(idx),bool),{})
        increments.append({'xyz':x[keep], 'rgb':rgb[keep], 'source_voxel':idx[keep],
                           'tree':cKDTree(x[keep]) if np.any(keep) else None})
        stats.append({'asset_index':i,'assigned_voxels':len(idx),'admitted_voxels':int(keep.sum()),**diagnostic})
    return increments, stats


def costs(obs, poses, history, increments, cfg, return_matches=False):
    n,m=len(obs['xyz']),len(history); costs=np.empty((n,m),np.float32)
    matches=[]; source=[]
    for i,h in enumerate(history):
        q=inverse(obs['xyz'],poses[i]); d,k=h['tree'].query(q,workers=1)
        value=np.minimum(d/cfg['distance_scale_m'],1)**2 + cfg['rgb_weight']*np.mean((obs['rgb']-h['rgb'][k])**2,axis=1)
        ref=h['xyz'][k].copy() if return_matches else None
        which=np.full(n,-1,np.int32) if return_matches else None
        if increments is not None and increments[i]['tree'] is not None:
            inc=increments[i]; ds,ks=inc['tree'].query(q,workers=1)
            ext=np.minimum(ds/cfg['distance_scale_m'],1)**2 + cfg['rgb_weight']*np.mean((obs['rgb']-inc['rgb'][ks])**2,axis=1)+cfg['extension_cost']
            choose=ext<value;value=np.minimum(value,ext)
            if return_matches:
                ref[choose]=inc['xyz'][ks[choose]];which[choose]=inc['source_voxel'][ks[choose]]
        # Semantic predictions are a soft penalty only; EVERY identity is evaluated.
        value+=cfg['semantic_weight']*(obs['class']!=h['class'])
        costs[:,i]=value
        if return_matches: matches.append(ref);source.append(which)
    labels=np.argmin(costs,axis=1);labels[np.min(costs,axis=1)>=cfg['null_cost']]=-1
    return costs, labels, matches, source


def refine(obs, labels, poses, history, increments, cfg):
    old,_,matches,_=costs(obs,poses,history,increments,cfg,True)
    proposed=[t.copy() for t in poses];accepted=0
    for i in range(len(history)):
        idx=np.flatnonzero((labels==i)&(old[:,i]<cfg['null_cost']))
        if len(idx)<6:continue
        # One rigid transform per asset, shared by every assigned fragment/point.
        weights=obs['mass'][idx]*(1-np.minimum(old[idx,i],1))**2
        if weights.sum()<=0:continue
        proposed[i]=fit_rigid(obs['xyz'][idx],matches[i][idx],weights)
    new,_,_,_=costs(obs,proposed,history,increments,cfg)
    for i in range(len(history)):
        idx=np.flatnonzero(labels==i)
        if len(idx) and np.dot(obs['mass'][idx],new[idx,i])<np.dot(obs['mass'][idx],old[idx,i])-1e-8:
            poses[i]=proposed[i];accepted+=1
    return poses,accepted


def run(obs, history, initial, cfg, method, monitor=None, native_labels=None):
    poses=[[t.copy() for t in pp] for pp in initial]
    labels=[a.copy() for a in native_labels] if native_labels is not None else [costs(o,p,history,None,cfg)[1] for o,p in zip(obs,poses)]
    initial_labels=[a.copy() for a in labels]
    rooted=method.startswith('rooted');bidirectional=method in ['joint_unrooted','rooted_joint']
    trace=[]
    # Forward methods freeze the source at the SAME initial labels and poses;
    # joint methods may revise source ownership/pose after receiving epoch 2.
    for it in range(cfg['iterations']):
        totalchange=0;accepted=0;source_stats=[]
        for t in ([0,1] if bidirectional else [1]):
            other=1-t
            inc=None
            if method!='frozen':
                inc,stat=make_surface(obs[other],labels[other],poses[other],history,cfg,rooted)
                source_stats.append({'source_epoch':other+1,'assets':stat})
            old=labels[t].copy()
            _,labels[t],_,_=costs(obs[t],poses[t],history,inc,cfg)
            poses[t],n=refine(obs[t],labels[t],poses[t],history,inc,cfg);accepted+=n
            _,labels[t],_,_=costs(obs[t],poses[t],history,inc,cfg)
            totalchange+=int(np.sum(old!=labels[t]))
        row={'iteration':it,'changed_voxels':totalchange,'accepted_pose_updates':accepted,
             'source_reassigned_voxels':int(np.sum(labels[0]!=initial_labels[0])), 'surface_stats':source_stats}
        trace.append(row)
        if monitor:monitor(row)
    return poses,labels,trace,initial_labels
