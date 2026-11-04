"""Direct finite-history factors, fragment sets, shared SE3 and sparse local motion.

Local experimental solver, not a calibrated posterior or originality claim.
No filesystem, simulator owner, physical ID or evaluation imports.
"""
from functools import lru_cache
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from comparative_evidence.estimation_v1 import reference_map, world_components
from inspection_decisions.projective_visibility_v1 import decode, component_mask, rasterize
from inspection_decisions.se3_visibility_v1 import register, meshes_from_history, valid_pose
from inspection_decisions.stable_branch_readout_v1 import readout
from inspection_decisions.correspondence_joint_v1 import assign
from scripts.joint_estimation import fit_rigid


def huber(r, transition=2.):
    r = np.abs(r)
    return np.where(r <= transition, r*r, 2*transition*r-transition**2)


def key_matrix(G):
    return tuple(np.asarray(G).ravel())


def union_of_masks(masks, assignment, ref):
    selected = [m for m, j in zip(masks, assignment) if j == ref]
    return np.logical_or.reduce(selected) if selected else np.zeros_like(masks[0], dtype=bool)


class DirectFactors:
    def __init__(self, raw, cfg):
        self.cfg = cfg; self.sensor = raw['sensor']; self.raw = raw
        self.history = [decode(o) for o in raw['history']]
        self.current = [decode(o) for o in raw['current']]
        self.refs = reference_map(self.history); self.n = len(self.refs)
        self.meshes = meshes_from_history(self.history, self.sensor, cfg)
        # Partition ONCE in the acquired local frame. Never re-merge after G.
        self.frames = [world_components(o) for o in self.current]
        self.points = [[c['points'][np.linspace(0, len(c['points'])-1,
                        min(cfg['point_samples'], len(c['points'])), dtype=int)] for c in cs] for cs in self.frames]
        self.masks = [[component_mask(o, c) for c in cs] for o, cs in zip(self.current, self.frames)]
        self.trees = [cKDTree(r['points']) for r in self.refs]
        self.evaluations = 0; self.render_calls = 0

    @lru_cache(maxsize=64)
    def archive_points(self, Gkey):
        inverse = np.linalg.inv(np.asarray(Gkey).reshape(4, 4))
        return [[p@inverse[:3, :3].T+inverse[:3, 3] for p in ps] for ps in self.points]

    @lru_cache(maxsize=512)
    def mask_view(self, f, selected):
        mask = np.logical_or.reduce([self.masks[f][i] for i in selected]) if selected else np.zeros_like(self.current[f]['depth'], dtype=bool)
        return mask, distance_transform_edt(~mask)

    @lru_cache(maxsize=256)
    def predicted(self, f, j, Gkey, dkey):
        G = np.asarray(Gkey).reshape(4, 4)
        T = np.linalg.inv(G)@self.current[f]['nominal_T']; self.render_calls += 1
        return rasterize(self.meshes[j], T, np.asarray(dkey), self.sensor).astype(np.float32)

    def point_cost(self, f, i, j, Gkey, delta):
        if self.frames[f][i]['group'] != self.refs[j]['group']:
            return 1e6
        q = self.archive_points(Gkey)[f][i]
        distance = self.trees[j].query(q-delta[j])[0]
        return float(np.mean(huber(distance/self.cfg['sigma'], self.cfg['huber_transition'])))

    def projective_factor(self, f, j, assignment, Gkey, delta):
        predicted = self.predicted(f, j, Gkey, tuple(delta[j]))
        mask, distance = self.mask_view(f, tuple(i for i, a in enumerate(assignment) if a == j))
        known = np.isfinite(predicted); count = int(known.sum())
        if not count:
            return (self.cfg['missing_support_penalty'] if mask.any() else 0.), 0
        observed = self.current[f]['depth']; overlap = known&mask
        foreground = known&~mask&(observed+self.cfg['occlusion_margin_m'] < predicted)
        edge = known&~mask&~foreground
        loss = np.zeros_like(predicted)
        loss[overlap] = np.minimum(huber((predicted[overlap]-observed[overlap])/self.cfg['ray_depth_sigma_m'],
                                       self.cfg['huber_transition']), self.cfg['missing_support_penalty'])
        if mask.any():
            loss[edge] = np.minimum(huber(distance[edge]/self.cfg['boundary_sigma_px'], self.cfg['huber_transition']),
                                   self.cfg['missing_support_penalty'])
        else:
            loss[edge] = self.cfg['missing_support_penalty']
        value = float(loss[known].sum()/count)
        if mask.any() and not overlap.any():
            value += self.cfg['missing_support_penalty']
        return value, int(overlap.sum())

    def global_ray(self, f, Gkey, delta):
        predicted = np.minimum.reduce([self.predicted(f, j, Gkey, tuple(delta[j])) for j in range(self.n)])
        known = np.isfinite(predicted)
        if not known.any():
            return self.cfg['missing_support_penalty'] if self.masks[f] else 0.
        observed = self.current[f]['depth']
        mask = np.logical_or.reduce(self.masks[f]) if self.masks[f] else np.zeros_like(known)
        foreground = known&~mask&(observed+self.cfg['occlusion_margin_m'] < predicted)
        comparable = known&~foreground
        # No target-mask identities in this ordinary scene-depth factor.
        r = (predicted[comparable]-observed[comparable])/self.cfg['ray_depth_sigma_m']
        return float(np.minimum(huber(r, self.cfg['huber_transition']),
                                self.cfg['missing_support_penalty']).sum()/max(int(known.sum()), 1))

    def energy(self, method, G, delta, assignment):
        self.evaluations += 1; cfg = self.cfg
        if not valid_pose(G, cfg) or np.max(abs(delta), initial=0) > cfg['max_local_motion_m']+1e-9:
            return float('inf')
        value = cfg['change_penalty']*int(np.sum(np.sum(delta**2, axis=1) > 1e-20))
        value += float(np.sum(delta**2)/cfg['motion_scale_m']**2)
        Gkey = key_matrix(G)
        for f, labels in enumerate(assignment):
            value += cfg['null_penalty']*sum(j < 0 for j in labels)
            if method.startswith('union'):
                value += sum(self.projective_factor(f, j, labels, Gkey, delta)[0] for j in range(self.n))
            elif method == 'point_set_ray':
                for j in range(self.n):
                    selected = [i for i, a in enumerate(labels) if a == j]
                    if selected:
                        q = np.concatenate([self.archive_points(Gkey)[f][i] for i in selected])
                        distance = self.trees[j].query(q-delta[j])[0]
                        value += float(np.mean(huber(distance/cfg['sigma'], cfg['huber_transition'])))
                value += self.global_ray(f, Gkey, delta)
            else:
                value += sum(self.point_cost(f, i, j, Gkey, delta) for i, j in enumerate(labels) if j >= 0)
        return float(value)

    def nearest_assignment(self, G, delta, one_to_one=False):
        labels = []; Gkey = key_matrix(G)
        for f, cs in enumerate(self.frames):
            C = np.array([[self.point_cost(f, i, j, Gkey, delta) for j in range(self.n)] for i in range(len(cs))])
            if one_to_one:
                pairs = assign(C, C < 1e5); a = [-1]*len(cs)
                for i, j in pairs: a[i] = j
            else:
                a = [int(np.argmin(row)) if row.min() < self.cfg['null_penalty'] else -1 for row in C]
            labels.append(a)
        return labels

    def proposals(self, G, delta, assignment):
        Gkey = key_matrix(G); sources = []; targets = []; weights = []; per_asset = [[] for _ in self.refs]
        for f, labels in enumerate(assignment):
            for i, j in enumerate(labels):
                if j < 0: continue
                q = self.archive_points(Gkey)[f][i]
                distances, nearest = self.trees[j].query(q-delta[j]); p = self.refs[j]['points'][nearest]
                w = 1/np.maximum(1., distances/self.cfg['sigma'])/len(q)
                sources.extend(p+delta[j]); targets.extend(self.points[f][i]); weights.extend(w)
                per_asset[j].append(np.sum((q-p)*w[:, None], axis=0)/max(w.sum(), 1e-12))
        pose = fit_rigid(np.asarray(sources), np.asarray(targets), np.asarray(weights)) if len(sources) >= 3 else G
        shifts = [np.mean(v, axis=0) if v else delta[j] for j, v in enumerate(per_asset)]
        return pose, np.clip(shifts, -self.cfg['max_local_motion_m'], self.cfg['max_local_motion_m'])


def optimize_mode(model, method, G0, warm_delta, branch_matches):
    cfg = model.cfg; G0 = np.asarray(G0)
    warm_delta = np.clip(warm_delta, -cfg['max_local_motion_m'], cfg['max_local_motion_m'])
    original = []
    for cs, pairs in zip(model.frames, branch_matches):
        labels = [-1]*len(cs)
        for i, j in pairs: labels[i] = j
        original.append(labels)
    one = method == 'point_one_to_one'; starts = []
    for delta in [np.zeros_like(warm_delta), warm_delta]:
        for labels in [original, model.nearest_assignment(G0, delta, one)]:
            starts.append((model.energy(method, G0, delta, labels), delta.copy(), [list(a) for a in labels]))
    score, delta, labels = min(starts, key=lambda x: x[0]); G = G0.copy()
    trace = [score]; updates = []; start_eval = model.evaluations
    def accept(candidate_G, candidate_delta, candidate_labels, kind):
        nonlocal G, delta, labels, score
        value = model.energy(method, candidate_G, candidate_delta, candidate_labels)
        if value < score-1e-9:
            G = candidate_G.copy(); delta = candidate_delta.copy(); labels = [list(a) for a in candidate_labels]
            score = value; trace.append(score); updates.append(kind)
            return True
        return False
    for round_index in range(cfg['joint_rounds']):
        # Direct point proposals are accepted ONLY by the method's full target.
        pose, shifts = model.proposals(G, delta, labels)
        if method != 'union_fixed_pose' and accept(pose, delta, labels, 'raw_point_pose_proposal'):
            _, shifts = model.proposals(G, delta, labels)
        for j in range(model.n):
            for d in [np.zeros(3), shifts[j]]:
                proposal = delta.copy(); proposal[j] = d
                accept(G, proposal, labels, 'shared_asset_motion_proposal')
        if one:
            accept(G, delta, model.nearest_assignment(G, delta, True), 'one_to_one_assignment')
        else:
            for f, cs in enumerate(model.frames):
                for i, c in enumerate(cs):
                    for j in [-1]+[j for j, ref in enumerate(model.refs) if ref['group'] == c['group']]:
                        if labels[f][i] == j: continue
                        proposal = [list(a) for a in labels]; proposal[f][i] = j
                        accept(G, delta, proposal, 'fragment_set_assignment')
        if method != 'union_fixed_pose':
            for axis in range(6):
                for sign in (-1, 1):
                    P = np.eye(4)
                    if axis < 3:
                        P[axis, 3] = sign*cfg['joint_translation_steps_m'][round_index]
                    else:
                        v = np.zeros(3); v[axis-3] = np.deg2rad(sign*cfg['joint_rotation_steps_deg'][round_index])
                        P[:3, :3] = Rotation.from_rotvec(v).as_matrix()
                    accept(P@G, delta, labels, 'direct_pose_coordinate')
        for j in range(model.n):
            if not any(j in a for a in labels): continue
            for axis in range(3):
                for sign in (-1, 1):
                    proposal = delta.copy(); proposal[j, axis] += sign*cfg['joint_motion_steps_m'][round_index]
                    accept(G, proposal, labels, 'direct_motion_coordinate')
    assert np.max(np.diff(trace), initial=0) <= 0
    if method == 'union_fixed_pose': np.testing.assert_array_equal(G, G0)
    return dict(G=G.tolist(), delta=delta.tolist(), assignment=labels, score=score,
        trace=trace, updates=updates, evaluations=model.evaluations-start_eval,
        initial_score=trace[0], initial_G=G0.tolist())


def prediction(model, result, method):
    G = np.asarray(result['G']); delta = np.asarray(result['delta']); labels = result['assignment']
    assignments = []; assets = []; Gkey = key_matrix(G)
    for f, a in enumerate(labels):
        for i, j in enumerate(a):
            if j >= 0:
                assignments.append(dict(ref_id=j, token=model.current[f]['token'], pixels=model.frames[f][i]['pixels'].tolist()))
    for j in range(model.n):
        support = 0
        for f, a in enumerate(labels):
            if method.startswith('union'):
                support += model.projective_factor(f, j, a, Gkey, delta)[1] if j in a else 0
            else:
                for i, asset in enumerate(a):
                    if asset == j:
                        q = model.archive_points(Gkey)[f][i]
                        support += int((model.trees[j].query(q-delta[j])[0] < .12).sum())
        label = int(np.max(abs(delta[j])) > .12) if support >= model.cfg['minimum_report_pixels'] else -1
        assets.append(dict(ref_id=j, label=label, delta=delta[j].tolist(),
            covariance=(np.eye(3)*.06**2).tolist(), current_support=support, association_margin_m=0.))
    return dict(assets=assets, references=[dict(ref_id=r['ref_id'], anchor_token=r['anchor_token'],
        anchor_pixels=r['anchor_pixels']) for r in model.refs], current_assignments=assignments,
        residual_median=None, delta_frame='archive', limits=['finite-history local experimental objective',
        'fixed covariance is not calibrated', 'map SE3; asset translations only; no global optimum guarantee'])


def run_methods(raw, cfg):
    started = time.perf_counter(); registration = register(raw, cfg)
    common = [readout(raw, np.asarray(b['G']), b['matches'], cfg) for b in registration['retained']]
    selected = min(range(len(common)), key=lambda k: common[k]['best']['score'])
    model = DirectFactors(raw, cfg)
    outputs = dict(legacy_compressed=dict(prediction=common[selected]['prediction'], selected=selected,
        scores=[c['best']['score'] for c in common], elapsed_s=time.perf_counter()-started))
    for method in cfg['methods']:
        if method == 'legacy_compressed': continue
        model.predicted.cache_clear(); model.archive_points.cache_clear(); model.mask_view.cache_clear()
        begin = time.perf_counter(); first_eval = model.evaluations; first_render = model.render_calls
        modes = []
        for c, b in zip(common, registration['retained']):
            G0 = np.asarray(c['map_transform_current_from_archive'])
            if not valid_pose(G0, cfg): G0 = np.asarray(b['G'])
            modes.append(optimize_mode(model, method, G0, c['best']['delta'], b['matches']))
        index = min(range(len(modes)), key=lambda k: modes[k]['score'])
        outputs[method] = dict(prediction=prediction(model, modes[index], method), selected=index,
            modes=modes, elapsed_s=time.perf_counter()-begin,
            evaluations=model.evaluations-first_eval, render_calls=model.render_calls-first_render)
    model.predicted.cache_clear(); model.archive_points.cache_clear(); model.mask_view.cache_clear()
    return dict(registration=registration, common_readouts=common, outputs=outputs,
        acquired_partition_counts=[len(cs) for cs in model.frames], total_elapsed_s=time.perf_counter()-started,
        limitations=['local mode search and coordinate optimization, no certified solution',
        'shared initials/rounds, different native factor computation; not equal FLOP',
        'fixed acquisition paths: no new active selection policy'])
