"""Observation-only bounded nuisance profiling for active identity/report evidence.

Not a calibrated sensor posterior or a novel generic VOI claim. All controls use
the same retained G/A/motion branches and the same forecast bank. Finite archive
patch occupancy is a forecast feature, NOT a declaration that unknown space is free.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from comparative_evidence.estimation_v1 import reference_map
from inspection_decisions.projective_visibility_v1 import decode, rasterize, triangles_from_mask
from comparative_evidence.scene import PALETTE
from inspection_decisions.se3_visibility_v1 import register, readout, meshes_from_history


def backend(raw, cfg):
    registration = register(raw, cfg)
    options = [readout(raw, np.asarray(b['G']), b['matches'], cfg)
               for b in registration['retained']]
    selected = min(range(len(options)), key=lambda k: options[k]['best']['score'])
    return dict(registration=registration, options=options, selected=selected,
                prediction=options[selected]['prediction'])


def report_distance(options, tolerance=.10):
    """Whole correlated branch consequences, in reference identities; no truth IDs."""
    n = len(options[0]['prediction']['assets']); K = len(options)
    maps = [{i: j for i, j in o['best']['matches'][0]} for o in options]
    result = np.zeros((K, K))
    for k in range(K):
        for l in range(k):
            wrong_identity = set()
            for i in set(maps[k]) | set(maps[l]):
                a, b = maps[k].get(i), maps[l].get(i)
                if a != b:
                    wrong_identity.update(j for j in (a, b) if j is not None)
            distance = 0
            for j, (a, b) in enumerate(zip(options[k]['prediction']['assets'],
                                           options[l]['prediction']['assets'])):
                differs = j in wrong_identity or a['label'] != b['label']
                if a['label'] == b['label'] == 1:
                    differs |= np.linalg.norm(np.asarray(a['delta'])-b['delta']) > tolerance
                distance += int(differs)
            result[k, l] = result[l, k] = distance/max(n, 1)
    return result


def feature(meshes, G, delta, nominal_T, sensor, cfg, perturb=None, foreground=None):
    # Perturbations are in current local-map coordinates, shared across assets.
    if perturb is not None:
        G = perturb @ G
    T = np.linalg.inv(G) @ nominal_T
    image = np.full((sensor['image_height'], sensor['image_width']), np.inf)
    for mesh, d in zip(meshes, delta):
        image = np.minimum(image, rasterize(mesh, T, d, sensor))
    if foreground is not None and len(foreground):
        blocker = rasterize(foreground, nominal_T, np.zeros(3), sensor)
        image = np.where(blocker+cfg['occlusion_margin_m'] < image, np.inf, image)
    mask = np.isfinite(image)
    support = gaussian_filter(mask.astype(float), cfg['feature_blur_px'])
    depth = gaussian_filter(np.where(mask, image, 0.), cfg['feature_blur_px'])
    depth /= cfg['feature_depth_scale_m']
    return np.r_[support.ravel(), depth.ravel()], int(mask.sum())


def bounded_profile(residual, jacobian, ridge=1e-6):
    """min_{|eta|<=1} ||r+J eta||^2 + ridge ||eta||^2; retains correlation."""
    residual = np.asarray(residual); jacobian = np.asarray(jacobian)
    raw = float(residual @ residual)
    if jacobian.shape[1] == 0 or raw == 0:
        return dict(raw=raw, profiled=raw, eta=[], success=True)
    A = np.vstack([jacobian, np.eye(jacobian.shape[1])*np.sqrt(ridge)])
    b = np.r_[-residual, np.zeros(jacobian.shape[1])]
    fit = lsq_linear(A, b, bounds=(-1., 1.), method='bvls', tol=1e-9, max_iter=100)
    eta = fit.x
    value = float(np.sum((residual+jacobian@eta)**2)+ridge*np.sum(eta**2))
    # Numerical fallback to feasible eta=0 cannot create artificial extra evidence.
    if value > raw:
        eta = np.zeros_like(eta); value = raw
    return dict(raw=raw, profiled=value, eta=eta.tolist(), success=bool(fit.success))


def expected_risk(distance, loss):
    """Uniform branch design weights; synthetic modal outcomes, NOT calibration."""
    K = len(distance)
    prior = np.full(K, 1/max(K, 1))
    entropy = 0.; risk = 0.
    for outcome in range(K):
        q = np.exp(-np.minimum(distance[:, outcome], 700.))*prior
        q /= q.sum()
        entropy += prior[outcome]*float(-np.sum(q*np.log(np.maximum(q, 1e-300))))
        risk += prior[outcome]*float(np.min(loss@q))
    return dict(entropy=entropy/max(np.log(K), 1.), report_risk=risk)


def forecast(raw, fitted, metadata, cfg):
    history = [decode(o) for o in raw['history']]
    meshes = meshes_from_history(history, raw['sensor'], cfg)
    # Observed non-palette foreground only; no unseen side or complete occluder.
    current = decode(raw['current'][0])
    distance = np.linalg.norm(current['rgb'].astype(float)[:, :, None]-PALETTE, axis=3).min(axis=2)
    foreground_mask = (distance > 30) & (current['depth'] < 8.)
    foreground = triangles_from_mask(current['depth'], current['nominal_T'],
                                     foreground_mask, raw['sensor'], cfg)
    sensor = dict(raw['sensor']); factor = cfg['forecast_downsample']
    for key in ('image_width', 'image_height'):
        sensor[key] //= factor
    sensor['focal_px'] /= factor
    options = fitted['options']; loss = report_distance(options)
    initial_refs = reference_map(history)
    rows = []; render_calls = 0
    for action in metadata['relative_actions']:
        a = action['id']; T = np.asarray(metadata['current'][a]['nominal_T'])
        bank = []; jacobians = []; coverage = []; known_pixels = []
        for o in options:
            G = np.asarray(o['map_transform_current_from_archive'])
            delta = np.asarray(o['best']['delta'])
            phi, pixels = feature(meshes, G, delta, T, sensor, cfg, foreground=foreground)
            render_calls += 1; columns = []
            for axis in range(6):
                P = np.eye(4)
                if axis < 3:
                    P[axis, 3] = cfg['nuisance_translation_m']
                else:
                    v = np.zeros(3); v[axis-3] = np.deg2rad(cfg['nuisance_rotation_deg'])
                    P[:3, :3] = Rotation.from_rotvec(v).as_matrix()
                plus, _ = feature(meshes, G, delta, T, sensor, cfg, P, foreground)
                minus, _ = feature(meshes, G, delta, T, sensor, cfg, np.linalg.inv(P), foreground)
                render_calls += 2; columns.append((plus-minus)/2)
            bank.append(phi); jacobians.append(np.stack(columns, axis=1)); known_pixels.append(pixels)
            # Common coverage control: references supported initially OR forecast
            # by a finite observed patch. Does not assume unobserved faces exist.
            covered = {j for pairs in o['best']['matches'] for _, j in pairs}
            camera = np.linalg.inv(G)@T
            blocker = rasterize(foreground, T, np.zeros(3), sensor)
            for j, mesh in enumerate(meshes):
                projected = rasterize(mesh, camera, delta[j], sensor)
                if (np.isfinite(projected)&~(blocker+cfg['occlusion_margin_m']<projected)).sum() >= 3:
                    covered.add(j)
            coverage.append(len(covered)/max(len(initial_refs), 1))
        K = len(bank); raw_distance = np.zeros((K, K)); profiled_distance = np.zeros((K, K)); profiles = []
        for k in range(K):
            for l in range(k):
                r = bank[k]-bank[l]; J = np.c_[jacobians[k], -jacobians[l]]
                p = bounded_profile(r, J, cfg['nuisance_ridge'])
                normalizer = max(len(r), 1)*cfg['feature_sigma']**2
                raw_distance[k, l] = raw_distance[l, k] = p['raw']/normalizer
                profiled_distance[k, l] = profiled_distance[l, k] = p['profiled']/normalizer
                profiles.append(dict(k=k, l=l, **p))
        ordinary = expected_risk(raw_distance, loss)
        candidate = expected_risk(profiled_distance, loss)
        base = np.asarray(metadata['current'][0]['nominal_T'])[:3, 3]
        travel = 2*float(np.linalg.norm(T[:3, 3]-base))
        seconds = travel/cfg['travel_speed_m_s']+raw['sensor']['acquisition_s']
        cost = cfg['resource_weight']*seconds
        rows.append(dict(action=a, name=action['name'], travel_with_return_m=travel,
            modeled_increment_s=seconds, resource_cost=cost, coverage=float(np.mean(coverage)),
            known_pixels=known_pixels, raw_distance=raw_distance.tolist(),
            profiled_distance=profiled_distance.tolist(), profiles=profiles,
            ordinary_information=ordinary['entropy']+cost,
            ordinary_report_risk=ordinary['report_risk']+cost,
            profiled_report_risk=candidate['report_risk']+cost,
            coverage_score=1-float(np.mean(coverage))+cost))
    return dict(actions=rows, report_distance=loss.tolist(), render_calls=render_calls,
                observed_foreground_triangles=len(foreground),
                branch_count=len(options), design_weights='uniform, not posterior',
                limits=['first-order bounded nuisance approximation, not nonlinear certificate',
                        'known-patch forecast features are not a complete RGBD likelihood',
                        'one-step modal-outcome approximation; no STOP or multi-step contribution'])


def choose(method, forecasts):
    if method == 'no_action':
        return 0
    if method == 'fixed_wide':
        return 2
    key = 'coverage_score' if method == 'coverage' else method
    return min(forecasts['actions'], key=lambda r: (r[key], r['action']))['action']
