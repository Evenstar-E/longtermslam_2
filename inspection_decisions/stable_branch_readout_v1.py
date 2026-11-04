"""Engineering adapter: a rigid frame change must not change component indices.

The old readout re-runs a coordinate-wise median fragment merge after a map
rotation. Coordinate medians are not rotation equivariant. Preserve the acquired
partition if that merge would change; otherwise execute the exact old code path.
"""
import numpy as np
from comparative_evidence.estimation_v1 import world_components, reference_map
from inspection_decisions.projective_visibility_v1 import decode, make_view, optimize
from inspection_decisions.se3_visibility_v1 import readout as legacy_readout, meshes_from_history
from inspection_decisions.correspondence_joint_v1 import pair_shift
from inspection_decisions.correspondence_motion_objective_v1 import solve as motion_solve


def transformed_groups(observation, inverse):
    result = []
    for c in world_components(observation):
        item = dict(c)
        item['points'] = c['points']@inverse[:3, :3].T+inverse[:3, 3]
        item['normals'] = c['normals']@inverse[:3, :3].T
        item['centre'] = np.median(item['points'], axis=0)
        result.append(item)
    return result


def partition_changed(observation, inverse):
    before = world_components(observation)
    corrected = dict(observation); corrected['nominal_T'] = inverse@observation['nominal_T']
    after = world_components(corrected)
    return [c['pixels'].tolist() for c in before] != [c['pixels'].tolist() for c in after]


def readout(raw, G, matches, cfg, projective=True):
    inverse = np.linalg.inv(G); current = [decode(o) for o in raw['current']]
    changed = [partition_changed(o, inverse) for o in current]
    if not any(changed):
        return legacy_readout(raw, G, matches, cfg, projective)
    history = [decode(o) for o in raw['history']]; refs = reference_map(history)
    grouped = [transformed_groups(o, inverse) for o in current]
    for o in current: o['nominal_T'] = inverse@o['nominal_T']
    frames = []; observations = [[] for r in refs]
    for f, (o, cs, pairs) in enumerate(zip(current, grouped, matches)):
        n = len(refs); D = np.zeros((len(cs), n, 3)); S = np.ones((len(cs), n))*1e4
        valid = np.zeros((len(cs), n), bool)
        for i, j in pairs:
            assert i < len(cs) and j < n
            c, r = cs[i], refs[j]
            if c['group'] != r['group']: continue
            D[i, j], S[i, j] = pair_shift(r['points'][::max(1, len(r['points'])//100)],
                                         c['points'][::max(1, len(c['points'])//100)])
            valid[i, j] = True; observations[j].append((f, i))
        frames.append(dict(token=o['token'], pixels=[c['pixels'].tolist() for c in cs],
                           D=D.tolist(), S=S.tolist(), valid=valid.tolist()))
    data = dict(references=[dict(ref_id=r['ref_id'], anchor_token=r['anchor_token'],
                                anchor_pixels=r['anchor_pixels']) for r in refs], frames=frames)
    fits = []
    if projective:
        meshes = meshes_from_history(history, raw['sensor'], cfg)
        views = [[make_view(o, c) for c in cs] for o, cs in zip(current, grouped)]
        for j, obs in enumerate(observations):
            if not obs or not len(meshes[j]): continue
            start = np.median([frames[f]['D'][i][j] for f, i in obs], axis=0)
            for f, i in obs:
                fit = optimize(meshes[j], [views[f][i]], start, raw['sensor'], cfg)
                frames[f]['D'][i][j] = fit['d']
                frames[f]['valid'][i][j] = sum(s['comparable_pixels'] for s in fit['stats']) >= cfg['minimum_comparable_pixels']
                fits.append(dict(reference=j, frame=f, component=i, fit=fit))
    result = motion_solve(data, 'shared_motion', cfg)
    residual = np.eye(4); residual[:3, 3] = result['best']['translation']
    result['map_transform_current_from_archive'] = (G@residual).tolist()
    result['initial_map_transform'] = G.tolist(); result['prediction']['delta_frame'] = 'archive'
    result['projective_fits'] = fits
    result['late_score'] = float(result['best']['score']+sum(r['fit']['loss'] for r in fits))
    result['engineering_partition_adapter'] = dict(changed_frames=changed,
        acquired_counts=[len(c) for c in grouped], reason='preserve acquired component indices under frame change')
    return result
