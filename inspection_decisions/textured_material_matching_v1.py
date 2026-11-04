"""Truth-blind, non-learned geometry/texture/material matching comparison.

No generator, private labels, frame groups, filenames or seeds enter the estimator.
Anonymous slots are spatial measurement groups, NOT persistent identities.
The visibility/triplet components are standard mechanisms, not a novelty claim.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


METHODS = ('geometry', 'texture', 'fusion', 'fusion_visibility',
           'fusion_persistence', 'persistent_visible')


def apply(T, x):
    return np.asarray(x) @ T[:3, :3].T + T[:3, 3]


def rigid(a, b, w=None):
    if len(a) < 3:
        return None
    w = np.ones(len(a)) if w is None else np.asarray(w)
    w = w / max(w.sum(), 1e-12)
    ac, bc = w @ a, w @ b
    u, s, vt = np.linalg.svd(((a-ac)*w[:, None]).T @ (b-bc))
    if s[1] < 1e-10:
        return None
    R = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, bc - R @ ac
    return T


def world_points(frame, uv):
    uv = np.asarray(uv)
    xy = np.rint(uv).astype(int)
    z = frame['depth'][xy[:, 1], xy[:, 0]]
    xyz = np.c_[(uv[:, 0]-frame['K'][0, 2])*z/frame['K'][0, 0],
                (uv[:, 1]-frame['K'][1, 2])*z/frame['K'][1, 1], z]
    return apply(frame['C'], xyz)


def extract_epoch(frames, cfg, seed):
    """Pool observations by spatial slot; cap features/points without frame IDs."""
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=cfg['sift_per_frame'], contrastThreshold=cfg['sift_contrast'])
    buckets = [dict(points=[], xyz=[], desc=[], response=[]) for _ in range(24)]
    for frame in frames:
        slots = frame['anonymous_slot']
        yy, xx = np.nonzero((slots >= 0) & frame['depth_valid'])
        pp = world_points(frame, np.c_[xx, yy])
        ids = slots[yy, xx]
        for i in np.unique(ids):
            buckets[int(i)]['points'].append(pp[ids == i])
        mask = ((slots >= 0) & frame['depth_valid']).astype(np.uint8)*255
        gray = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, mask)
        if desc is None:
            continue
        uv = np.array([k.pt for k in kp])
        pix = np.rint(uv).astype(int)
        pix[:, 0] = np.clip(pix[:, 0], 0, slots.shape[1]-1)
        pix[:, 1] = np.clip(pix[:, 1], 0, slots.shape[0]-1)
        ids = slots[pix[:, 1], pix[:, 0]]
        good = (ids >= 0) & frame['depth_valid'][pix[:, 1], pix[:, 0]]
        xyz = world_points(frame, pix[good])
        dd = np.sqrt(desc[good] / np.maximum(desc[good].sum(1, keepdims=True), 1e-12))
        response = np.array([k.response for k in kp])[good]
        for i in np.unique(ids[good]):
            take = ids[good] == i
            b = buckets[int(i)]
            b['xyz'].append(xyz[take]); b['desc'].append(dd[take]); b['response'].append(response[take])
    out = []
    rng = np.random.default_rng(seed)
    for b in buckets:
        p = np.concatenate(b['points']) if b['points'] else np.zeros((0, 3))
        if len(p):
            _, ix = np.unique(np.floor(p/cfg['voxel_m']).astype(int), axis=0, return_index=True)
            p = p[ix]
            if len(p) > cfg['points_per_slot']:
                p = p[np.sort(rng.choice(len(p), cfg['points_per_slot'], replace=False))]
        if b['xyz']:
            xyz = np.concatenate(b['xyz']); desc = np.concatenate(b['desc']); resp = np.concatenate(b['response'])
            # One physical spatial cell per feature: suppress duplicated SIFT orientations.
            order = np.argsort(-resp, kind='stable')
            _, unique = np.unique(np.floor(xyz[order]/cfg['voxel_m']).astype(int), axis=0, return_index=True)
            ix = order[np.sort(unique)[:cfg['features_per_slot']]]
            xyz, desc = xyz[ix], desc[ix]
        else:
            xyz, desc = np.zeros((0, 3)), np.zeros((0, 128), np.float32)
        out.append(dict(points=p, xyz=xyz, desc=desc,
                        tree=cKDTree(p) if len(p) else None))
    return out


def matches(a, b, cfg):
    if min(len(a['desc']), len(b['desc'])) < 2:
        return dict(ia=np.zeros(0, int), ib=np.zeros(0, int), distance=np.zeros(0))
    d = np.maximum(0., 2.-2.*a['desc'] @ b['desc'].T)
    order = np.argsort(d, axis=1)[:, :2]
    ix = np.arange(len(d)); first, second = order[:, 0], order[:, 1]
    good = ((d[ix, first] < cfg['ratio']**2*d[ix, second]) &
            (d[ix, first] < cfg['descriptor_max']**2) & (d.argmin(0)[first] == ix))
    return dict(ia=ix[good], ib=first[good], distance=np.sqrt(d[ix[good], first[good]]))


def icp(a, b, T, cfg):
    if not len(a['points']) or b['tree'] is None:
        return T.copy()
    T = T.copy()
    for _ in range(cfg['icp_iterations']):
        p = apply(T, a['points']); dist, ix = b['tree'].query(p)
        take = dist <= max(.015, np.quantile(dist, .75))
        step = rigid(p[take], b['points'][ix[take]])
        if step is None:
            break
        T = step @ T
        if np.linalg.norm(step-np.eye(4)) < 1e-6:
            break
    return T


def geometry_loss(a, b, T, cfg):
    if a['tree'] is None or b['tree'] is None:
        return 2.
    da = b['tree'].query(apply(T, a['points']))[0]
    db = a['tree'].query(apply(np.linalg.inv(T), b['points']))[0]
    return .5*(np.minimum(da/cfg['geometry_scale_m'], 1.).mean() +
               np.minimum(db/cfg['geometry_scale_m'], 1.).mean())


def texture_loss(a, b, match, T, cfg, eligible=None):
    if not len(match['ia']):
        return 1.
    residual = np.linalg.norm(apply(T, a['xyz'][match['ia']])-b['xyz'][match['ib']], axis=1)
    support = np.exp(-.5*(residual/cfg['material_scale_m'])**2)
    support *= np.exp(-.5*(match['distance']/.5)**2)
    if eligible is not None:
        support *= eligible
    return 1.-min(1., support.sum()/cfg['positive_support'])


def ray_state(points, frames, target_slot, cfg):
    """0 unknown/occluded, 1 observed surface, 2 contradicted known free space.

Use the best supporting view. A point behind a nearer return is censored, never
turned into a target correspondence. Invalid depth and outside are unknown.
"""
    state = np.zeros(len(points), np.int8)
    gap = np.full(len(points), np.inf)
    for f in frames:
        pc = apply(np.linalg.inv(f['C']), points)
        uv = pc[:, :2]/np.maximum(pc[:, 2:3], 1e-8)
        uv = uv*np.array([f['K'][0, 0], f['K'][1, 1]])+f['K'][:2, 2]
        xy = np.rint(uv).astype(int)
        h, w = f['depth'].shape
        good = (pc[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        ix = np.flatnonzero(good)
        if not len(ix):
            continue
        x, y = xy[ix].T
        valid = f['depth_valid'][y, x]
        ix, x, y = ix[valid], x[valid], y[valid]
        dz = f['depth'][y, x]-pc[ix, 2]
        owned = f['anonymous_slot'][y, x] == target_slot
        hit = owned & (np.abs(dz) <= cfg['visibility_tolerance_m'])
        free = dz > cfg['visibility_tolerance_m']
        update = hit & (np.abs(dz) < gap[ix])
        state[ix[update]], gap[ix[update]] = 1, np.abs(dz[update])
        update = free & (state[ix] != 1) & (dz < gap[ix])
        state[ix[update]], gap[ix[update]] = 2, dz[update]
    return state, gap


def visibility_loss(a, b, T, frames_a, frames_b, slot_a, slot_b, cfg):
    sa, ga = ray_state(apply(T, a['points']), frames_b, slot_b, cfg)
    sb, gb = ray_state(apply(np.linalg.inv(T), b['points']), frames_a, slot_a, cfg)
    s, g = np.r_[sa, sb], np.r_[ga, gb]
    exposed = s != 0
    if not exposed.any():
        return 1.
    loss = np.minimum(g[exposed]/cfg['geometry_scale_m'], 1.).mean()
    # No free benefit for hiding every point: require measured positive support.
    deficit = max(0., 1.-np.sum(s == 1)/cfg['positive_support'])
    return min(2., loss+deficit)


def ransac(a, b, match, cfg, seed):
    n = len(match['ia'])
    if n < 3:
        return None
    p, q = a['xyz'][match['ia']], b['xyz'][match['ib']]
    rng = np.random.default_rng(seed)
    best, best_value = None, -1.
    for _ in range(cfg['ransac_iterations']):
        ix = rng.choice(n, 3, replace=False)
        T = rigid(p[ix], q[ix])
        if T is None:
            continue
        d = np.linalg.norm(apply(T, p)-q, axis=1)
        value = np.exp(-.5*(d/cfg['material_scale_m'])**2).sum()
        if value > best_value:
            best, best_value = T, value
    if best is not None:
        d = np.linalg.norm(apply(best, p)-q, axis=1)
        take = d < cfg['material_scale_m']*2
        refined = rigid(p[take], q[take])
        if refined is not None:
            best = refined
    return best


def geometric_pools(epochs, cfg):
    pools = {}
    for e, f in [(0, 1), (0, 2), (1, 2)]:
        v = []
        for a in epochs[e]:
            row = []
            for b in epochs[f]:
                T = np.eye(4)
                if len(a['points']) and len(b['points']):
                    T[:3, 3] = np.median(b['points'], axis=0)-np.median(a['points'], axis=0)
                row.append([T, icp(a, b, T, cfg)])
            v.append(row)
        pools[e, f] = v
    return pools


def score_transform(method, a, b, match, T, frames_a, frames_b, i, j, cfg):
    geom = geometry_loss(a, b, T, cfg)
    if method == 'geometry':
        return geom
    if method == 'texture':
        return texture_loss(a, b, match, T, cfg)
    if method in ('fusion_visibility', 'persistent_visible'):
        vis = visibility_loss(a, b, T, frames_a, frames_b, i, j, cfg)
        elig = None
        if len(match['ia']):
            s, _ = ray_state(apply(T, a['xyz'][match['ia']]), frames_b, j, cfg)
            elig = s == 1
        return .5*geom+.5*vis+texture_loss(a, b, match, T, cfg, elig)
    return geom+texture_loss(a, b, match, T, cfg)


def solve_assignments(costs, cfg, triple=None):
    # Coordinate Hungarian updates with both chronological starts, same for all.
    def perm(c):
        i, j = linear_sum_assignment(c)
        p = np.empty(24, int); p[i] = j
        return p
    starts = [(perm(costs[0, 1]), perm(costs[0, 2]))]
    p1 = starts[0][0]
    starts.append((p1.copy(), perm(costs[1, 2])[p1]))
    best = None
    for p1, p2 in starts:
        for _ in range(cfg['assignment_iterations']):
            c1 = costs[0, 1]+cfg['cycle_weight']*costs[1, 2][:, p2].T
            if triple is not None:
                c1 = c1+cfg['persistence_weight']*triple[np.arange(24)[:, None], np.arange(24)[None, :], p2[:, None]]
            p1 = perm(c1)
            c2 = costs[0, 2]+cfg['cycle_weight']*costs[1, 2][p1, :]
            if triple is not None:
                c2 = c2+cfg['persistence_weight']*triple[np.arange(24)[:, None], p1[:, None], np.arange(24)[None, :]]
            p2 = perm(c2)
        value = (costs[0, 1][np.arange(24), p1]+costs[0, 2][np.arange(24), p2]+
                 cfg['cycle_weight']*costs[1, 2][p1, p2]).sum()
        if triple is not None:
            value += cfg['persistence_weight']*triple[np.arange(24), p1, p2].sum()
        if best is None or value < best[0]:
            best = value, p1.copy(), p2.copy()
    return best[1:]


def persistence_loss(a, b, c, ab, ac, T1, T2, cfg):
    common, i, j = np.intersect1d(ab['ia'], ac['ia'], return_indices=True)
    if not len(common):
        return 1., 0
    p = a['xyz'][common]
    q = apply(np.linalg.inv(T1), b['xyz'][ab['ib'][i]])
    r = apply(np.linalg.inv(T2), c['xyz'][ac['ib'][j]])
    d = np.sqrt((np.sum((p-q)**2, axis=1)+np.sum((p-r)**2, axis=1)+np.sum((q-r)**2, axis=1))/3.)
    mass = np.exp(-.5*(d/cfg['material_scale_m'])**2).sum()
    return 1.-min(1., mass/cfg['positive_support']), len(common)


def compare(epochs, frames, cfg, geom_pools, seed, progress=None):
    pools, mm = {}, {}
    costs = {m:{} for m in METHODS}
    single = {m:{} for m in METHODS}
    for e, f in [(0, 1), (0, 2), (1, 2)]:
        pools[e, f], mm[e, f] = [], []
        for m in METHODS:
            costs[m][e, f] = np.zeros((24, 24))
            single[m][e, f] = np.zeros((24, 24, 4, 4))
        for i, a in enumerate(epochs[e]):
            poolrow, matchrow = [], []
            for j, b in enumerate(epochs[f]):
                match = matches(a, b, cfg)
                T = ransac(a, b, match, cfg, seed+e*10000+f*1000+i*24+j)
                # All fusion variants have EXACTLY these same proposed poses.
                pool = list(geom_pools[e, f][i][j])
                if T is not None:
                    pool += [T, icp(a, b, T, cfg)]
                poolrow.append(pool); matchrow.append(match)
                for m in ('geometry', 'texture', 'fusion', 'fusion_visibility'):
                    allowed = pool[:2] if m == 'geometry' else ([T if T is not None else np.eye(4)] if m == 'texture' else pool)
                    values = [score_transform(m, a, b, match, t, frames[e], frames[f], i, j, cfg) for t in allowed]
                    k = int(np.argmin(values))
                    costs[m][e, f][i, j], single[m][e, f][i, j] = values[k], allowed[k]
                for m, base in [('fusion_persistence', 'fusion'), ('persistent_visible', 'fusion_visibility')]:
                    costs[m][e, f][i, j] = costs[base][e, f][i, j]
                    single[m][e, f][i, j] = single[base][e, f][i, j]
            pools[e, f].append(poolrow); mm[e, f].append(matchrow)
        if progress:
            progress(f'pair {e}-{f} complete')
    predictions = {}
    for m in METHODS:
        triple = None
        if m in ('fusion_persistence', 'persistent_visible'):
            triple = np.ones((24, 24, 24))
            for i in range(24):
                for j in range(24):
                    for k in range(24):
                        triple[i, j, k] = persistence_loss(epochs[0][i], epochs[1][j], epochs[2][k],
                            mm[0, 1][i][j], mm[0, 2][i][k], single[m][0, 1][i,j], single[m][0,2][i,k], cfg)[0]
        p1, p2 = solve_assignments(costs[m], cfg, triple)
        transforms, track_counts, objectives = [], [], []
        for i, (j, k) in enumerate(zip(p1, p2)):
            a, b, c = epochs[0][i], epochs[1][j], epochs[2][k]
            ab, ac, bc = mm[0, 1][i][j], mm[0, 2][i][k], mm[1, 2][j][k]
            # Same finite joint-pose budget for all fusion controls, using actual
            # data-fitted continuous poses; no catalogue or ground-truth poses.
            ps1 = pools[0, 1][i][j] if m.startswith('fusion') or m == 'persistent_visible' else [single[m][0, 1][i, j]]
            ps2 = pools[0, 2][i][k] if m.startswith('fusion') or m == 'persistent_visible' else [single[m][0, 2][i, k]]
            best = None
            for T1 in ps1:
                for T2 in ps2:
                    vals = [score_transform(m, a, b, ab, T1, frames[0], frames[1], i, j, cfg),
                            score_transform(m, a, c, ac, T2, frames[0], frames[2], i, k, cfg)]
                    T12 = T2 @ np.linalg.inv(T1)
                    value = sum(vals)+cfg['cycle_weight']*score_transform(m, b, c, bc, T12, frames[1], frames[2], j, k, cfg)
                    pl, count = persistence_loss(a, b, c, ab, ac, T1, T2, cfg)
                    if m in ('fusion_persistence', 'persistent_visible'):
                        value += cfg['persistence_weight']*pl
                    if best is None or value < best[0]:
                        best = value, T1.copy(), T2.copy(), count
            objectives.append(best[0]); transforms.append([best[1], best[2]]); track_counts.append(best[3])
        predictions[m] = dict(target_slots=np.c_[p1, p2], transforms=np.array(transforms),
                              triplet_counts=np.array(track_counts), objectives=np.array(objectives))
    return predictions, dict(features=[[len(x['desc']) for x in ep] for ep in epochs],
                             common_fusion_pool_sizes={f'{e}-{f}':[[len(v) for v in row] for row in mat] for (e,f),mat in pools.items()},
                             matches={f'{e}-{f}':[[len(v['ia']) for v in row] for row in mat] for (e,f),mat in mm.items()})
