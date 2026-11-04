"""Fresh physical RGBD, costed relative actions, shared-backend policy experiment."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.scene import vertices, halfspaces, render
from comparative_evidence.interface import Episode, shortest_path
from comparative_evidence.estimation_v1 import reference_map, world_components
from comparative_evidence.evaluation_v3 import evaluate
from scripts.partial_surface_scene import camera
from inspection_decisions.projective_visibility_v1 import capture, decode
from inspection_decisions.profiled_identity_actions_v1 import backend, forecast, choose

CFG = 'configs/profiled_identity_actions_v1.json'
CONTRACT = 'research/profiled_identity_actions_contract_v1_2026-10-23.md'
SOURCES = [CFG, CONTRACT, __file__, 'inspection_decisions/profiled_identity_actions_v1.py',
    'inspection_decisions/se3_visibility_v1.py', 'inspection_decisions/projective_visibility_v1.py',
    'inspection_decisions/correspondence_joint_v1.py', 'inspection_decisions/correspondence_motion_objective_v1.py',
    'comparative_evidence/estimation_v1.py', 'comparative_evidence/scene.py',
    'comparative_evidence/interface.py', 'comparative_evidence/evaluation_v1.py',
    'comparative_evidence/evaluation_v3.py', 'scripts/partial_surface_scene.py',
    'scripts/joint_estimation.py', 'configs/se3_visibility_v1.json',
    'configs/comparative_evidence_interface_v1.json', 'tests/test_profiled_identity_actions_v1.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default,
                                    allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merged_config():
    cfg = read('configs/se3_visibility_v1.json'); cfg.update(read(CFG))
    return cfg


def relative_motion(action):
    P = np.eye(4); P[:3, 3] = action['body_translation_m']
    P[:3, :3] = Rotation.from_euler('y', action['body_yaw_deg'], degrees=True).as_matrix()
    return P


def generate(root, world, cfg, sensor):
    root.mkdir(parents=True); (root/'payloads').mkdir(); (root/'evaluation').mkdir()
    rng = np.random.default_rng(cfg['seed_start']+world)
    centres = np.array([[(j % 4-1.5)*1.65, (j//4-.5)*1.65, 5.5] for j in range(8)])
    centres += rng.normal(0, cfg['layout_jitter_m'], (8, 3))
    past_vertices = [[p*.75+c for p in vertices('hex_prism')] for c in centres]
    delta = np.zeros((8, 3))
    if world >= cfg['static_worlds']:
        ids = rng.choice(8, cfg['changed_assets'], replace=False)
        direction = rng.normal(size=(len(ids), 3)); direction /= np.linalg.norm(direction, axis=1)[:, None]
        delta[ids] = direction*rng.uniform(cfg['motion_min_m'], cfg['motion_max_m'], (len(ids), 1))
    current_vertices = [[p+d for p in parts] for parts, d in zip(past_vertices, delta)]
    past = [halfspaces(p) for p in past_vertices]; current = [halfspaces(p) for p in current_vertices]
    groups = np.zeros(8, int); occluders = []; occluder_vertices = []
    if world % 3 == cfg['occluder_world_mod3']:
        sign = rng.choice([-1., 1.])
        v = np.array([[sign*x, y, z] for x in (.28, .62) for y in (-1.8, 1.8) for z in (3.9, 4.05)])
        occluder_vertices.append(v); occluders.append(halfspaces([v]))
    G = np.eye(4); G[:3, :3] = Rotation.from_euler('xyz', rng.uniform([-2., -2., 5.], [2., 2., 9.]), degrees=True).as_matrix()
    G[:3, 3] = rng.uniform(-.5, .5, 3)
    T0 = camera(np.array([0., 0., 2.]), np.array([0., 0., 5.5]))
    Ts = [T0]+[T0@relative_motion(a) for a in cfg['actions']]
    H = camera(np.array([0., 0., -1.5]), np.array([0., 0., 5.5]))
    catalog = dict(sensor=sensor, history=[], current=[], edges=[], relative_actions=cfg['actions'],
        error_model=dict(map_registration_prior=dict(translation_infinity_bound_m=cfg['se3_translation_bound_m'],
                                                   rotation_angle_bound_deg=cfg['se3_rotation_bound_deg'])),
        simulation_limits=['palette foreground extraction', 'translation-only object changes',
                           'discrete relative camera actions; no UAV dynamics'])
    poses = {}
    for epoch, cameras, parts in [('history', [H], past), ('current', Ts, current)]:
        for view, T in enumerate(cameras):
            rgb, depth, owner = render(parts, T, groups, sensor, rng, occluders if epoch == 'current' else ())
            nominal = G@T if epoch == 'current' else T
            token = f'{epoch}_{view:02d}'
            np.savez_compressed(root/'payloads'/f'{token}.npz', rgb=rgb, depth=depth, nominal_T=nominal)
            np.savez_compressed(root/'evaluation'/f'{token}_owner.npz', owner=owner)
            catalog[epoch].append(dict(id=view, token=token, nominal_T=nominal.tolist(), nuisance_group=epoch))
            poses[token] = T.tolist()
    # Generator-only conservative half-space certificate: all paths remain on
    # the camera side of the nearest true geometry, including the occluder.
    cloud = np.concatenate([p for parts in current_vertices for p in parts]+occluder_vertices)
    minimum_geometry_z = float(cloud[:, 2].min())
    minimum_clearance = float(minimum_geometry_z-max(T[2, 3] for T in Ts))
    assert minimum_clearance > sensor['clearance_m']
    for action in cfg['actions']:
        a = action['id']; length = float(np.linalg.norm(Ts[a][:3, 3]-T0[:3, 3]))
        catalog['edges'].append(dict(a=0, b=a, distance_m=length))
    save(root/'catalog.json', catalog)
    save(root/'evaluation/truth.json', dict(seed=cfg['seed_start']+world, session_G=G,
        centres=centres, delta=delta, changed=np.linalg.norm(delta, axis=1) > 0,
        poses=poses, historical_vertices=past_vertices, current_vertices=current_vertices,
        occluder_vertices=occluder_vertices, path_minimum_z_clearance_m=minimum_clearance))


def acquired(root, action, sensor):
    episode = Episode(root, 1, 1); history = [episode.act('retrieve_history', 0)]
    current = [episode.preview()]
    if action:
        current.append(episode.act('observe_current', int(action)))
    raw = capture(history, current, sensor)
    return episode, raw


def identity_diagnostics(root, raw, fitted):
    def owners(obs, components):
        with np.load(root/'evaluation'/f"{obs['token']}_owner.npz") as z:
            image = z['owner'].ravel()
        result = []
        for c in components:
            labels = image[c['pixels']]; labels = labels[labels >= 0]
            result.append(int(np.bincount(labels, minlength=8).argmax()) if len(labels) else -1)
        return result
    hs = [decode(o) for o in raw['history']]; refs = reference_map(hs)
    with np.load(root/'evaluation'/f"{hs[0]['token']}_owner.npz") as z:
        image = z['owner'].ravel()
    ref_ids = []
    for ref in refs:
        values = image[ref['anchor_pixels']]; values = values[values >= 0]
        ref_ids.append(int(np.bincount(values, minlength=8).argmax()))
    current = [decode(o) for o in raw['current']]
    cur_ids = [owners(o, world_components(o)) for o in current]
    def check_branch(branch):
        pairs = [(cur_ids[f][i], ref_ids[j]) for f, matches in enumerate(branch['matches']) for i, j in matches]
        return dict(pairs=pairs, correct=sum(a == b for a, b in pairs), total=len(pairs),
                    all_supported_pairs_correct=bool(pairs) and all(a == b for a, b in pairs))
    registration = fitted['registration']
    return dict(reference_owners=ref_ids, current_owners=cur_ids,
        distinct_retained_assignments=len({str(b['matches']) for b in registration['retained']}),
        retained=[check_branch(b) for b in registration['retained']],
        correct_branch_generated=any(check_branch(b)['all_supported_pairs_correct'] for b in registration['solutions']),
        correct_branch_retained=any(check_branch(b)['all_supported_pairs_correct'] for b in registration['retained']))


def objective(metric, cost, cfg, acquisition_s):
    increment = cost['travel_m']/cfg['travel_speed_m_s']+cost['modeled_acquisition_s']-acquisition_s
    return metric['task_loss']/8+cfg['resource_weight']*increment


def run():
    cfg = merged_config(); out = Path(cfg['output']); assert not out.exists(), 'Sealed output exists'
    out.mkdir(parents=True)
    sources = {str(p): digest(p) for p in SOURCES}
    save(out/'frozen_contract.json', dict(config=cfg, sources=sources, frozen_before_world_generation=True,
        status='fresh exploratory experiment; not confirmation; no outcome-tuned parameters'))
    sensor = read('configs/comparative_evidence_interface_v1.json'); sensor['factors'] = {}
    rows = []; mechanism = []; start = time.perf_counter(); actual_backend_calls = 0; api_paths = 0
    for world in range(cfg['worlds']):
        wd = out/f'{world:04d}'; wd.mkdir(); root = out/'scenes'/f'{world:04d}'
        generate(root, world, cfg, sensor)
        episode, initial = acquired(root, 0, sensor); api_paths += 1
        save(wd/'initial_observed.json', initial)
        fitted = backend(initial, cfg); actual_backend_calls += 1
        save(wd/'initial_backend.json', fitted)
        forecasts = forecast(initial, fitted, episode.metadata(), cfg)
        decisions = {method: choose(method, forecasts) for method in cfg['methods']}
        # All forecast scores and policy actions persist before evaluator access.
        save(wd/'forecasts.json', forecasts); save(wd/'decisions.json', decisions)
        cost = episode.finish(); save(wd/'endpoint_0_cost.json', dict(cost=cost, events=episode.state()['events']))
        metric = evaluate(root, fitted['prediction'])
        save(wd/'endpoint_0_evaluation.json', metric)
        diagnostics = identity_diagnostics(root, initial, fitted)
        save(wd/'initial_identity_diagnostic.json', diagnostics)
        mechanism.append(dict(world=world, changed=world >= cfg['static_worlds'], **diagnostics))
        endpoints = {0: dict(metric=metric, cost=cost)}
        # Counterfactual endpoints acquired by legal API; never exposed to choose().
        for action in cfg['actions']:
            a = action['id']; ep, raw = acquired(root, a, sensor); api_paths += 1
            save(wd/f'endpoint_{a}_observed.json', raw)
            result = backend(raw, cfg); actual_backend_calls += 1
            save(wd/f'endpoint_{a}_backend.json', result)
            action_cost = ep.finish()
            save(wd/f'endpoint_{a}_cost.json', dict(cost=action_cost, events=ep.state()['events']))
            action_metric = evaluate(root, result['prediction'])
            save(wd/f'endpoint_{a}_evaluation.json', action_metric)
            save(wd/f'endpoint_{a}_identity_diagnostic.json', identity_diagnostics(root, raw, result))
            endpoints[a] = dict(metric=action_metric, cost=action_cost)
        best = min(objective(e['metric'], e['cost'], cfg, sensor['acquisition_s'])
                   for a, e in endpoints.items() if a != 0)
        for method, a in decisions.items():
            # Execute the selected trajectory afresh. Exact input match permits
            # common backend memoization; no unacquired view enters this policy.
            ep, raw = acquired(root, a, sensor); api_paths += 1
            assert raw == read(wd/('initial_observed.json' if a == 0 else f'endpoint_{a}_observed.json'))
            action_cost = ep.finish(); endpoint = endpoints[a]
            for key in ('travel_m', 'additional_current_views', 'modeled_acquisition_s', 'returned'):
                assert action_cost[key] == endpoint['cost'][key]
            value = objective(endpoint['metric'], action_cost, cfg, sensor['acquisition_s'])
            record = dict(world=world, method=method, action=a, changed=world >= cfg['static_worlds'],
                metric=endpoint['metric'], cost=action_cost, objective=value,
                one_action_hindsight_regret=value-best if a else None,
                common_backend_memoized=True, common_forecast_render_calls=forecasts['render_calls'])
            save(wd/f'policy_{method}.json', dict(**record, events=ep.state()['events']))
            rows.append(record)
        save(out/'progress.json', dict(completed_worlds=world+1, total_worlds=cfg['worlds'],
             elapsed_s=time.perf_counter()-start, backend_calls=actual_backend_calls, api_paths=api_paths))
        print('WORLD', world, 'A', diagnostics['distinct_retained_assignments'],
              'correct retained', diagnostics['correct_branch_retained'], 'decisions', decisions,
              'losses', {a: round(e['metric']['task_loss'], 3) for a, e in endpoints.items()}, flush=True)
    assert all(digest(p) == sha for p, sha in sources.items()), 'Frozen execution source changed'
    save(out/'rows.json', rows); save(out/'mechanism.json', mechanism)
    summary = []
    for scope in ('all', 'changed', 'static'):
        for method in cfg['methods']:
            subset = [r for r in rows if r['method'] == method and
                      (scope == 'all' or r['changed'] == (scope == 'changed'))]
            summary.append(dict(scope=scope, method=method, worlds=len(subset),
                qualified_tp=sum(r['metric']['qualified_tp'] for r in subset),
                true_changes=sum(r['metric']['changed'] for r in subset),
                fp=sum(r['metric']['fp'] for r in subset),
                bad_motion=sum(r['metric']['bad_motion_reports'] for r in subset),
                wrong_known=sum(r['metric']['wrong_known'] for r in subset),
                known=sum(r['metric']['known'] for r in subset),
                mean_task_loss=float(np.mean([r['metric']['task_loss'] for r in subset])),
                mean_travel_m=float(np.mean([r['cost']['travel_m'] for r in subset])),
                mean_objective=float(np.mean([r['objective'] for r in subset])),
                returned=all(r['cost']['returned'] for r in subset)))
    save(out/'summary.json', dict(rows=summary, worlds=cfg['worlds'], backend_calls=actual_backend_calls,
        API_paths=api_paths, policy_paths=len(rows), raw_RGBD_payloads=8*cfg['worlds'],
        elapsed_s=time.perf_counter()-start, exclusions=[], confirmation=False, innovation_confirmed=False))
    print('DONE', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run'])
    parser.parse_args(); run()
