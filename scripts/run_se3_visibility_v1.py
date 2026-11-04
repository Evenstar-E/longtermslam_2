"""SE3 coordinate-frame paired pilot with actual RGBD and restricted history."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture
from inspection_decisions.se3_visibility_v1 import run_methods
from inspection_decisions.graph_dataset import digest
from scripts.run_correspondence_joint_pilot_v1 import generate
from scripts.run_evidence_value_pilot_v1 import read,save

CFG='configs/se3_visibility_v1.json'
CONTRACT='research/se3_visibility_contract_v1_2026-10-23.md'


def make_G(seed,cfg):
    rng=np.random.default_rng(seed+1901);G=np.eye(4);angles=[*rng.uniform(-cfg['injected_roll_pitch_deg'],cfg['injected_roll_pitch_deg'],2),rng.uniform(*cfg['injected_yaw_deg'])*(1 if seed%2 else -1)]
    G[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix();G[:3,3]=rng.uniform(-cfg['injected_translation_bound_m'],cfg['injected_translation_bound_m'],3)
    return G


def scene_root(out,family,variant):return out/'scenes'/f'world_{family:02d}_{variant}'


def generate_se3(root,seed,repeated,G,cfg):
    generate(root,seed,'repeat_occluded' if repeated else 'normal',2 if (seed-cfg['seed_start'])%2 else 0,cfg)
    catalog=read(root/'catalog.json');truth=read(root/'evaluation/truth.json');oldshift=np.array(truth.pop('injected_session_translation'));base=[]
    for row in catalog['current']:
        path=root/'payloads'/f"{row['token']}.npz"
        with np.load(path) as z:rgb=z['rgb'].copy();depth=z['depth'].copy();T=z['nominal_T'].copy()
        T[:3,3]-=oldshift;base.append(T.tolist());T=G@T
        np.savez_compressed(path,rgb=rgb,depth=depth,nominal_T=T);row['nominal_T']=T.tolist();row['pose_frame']='current_local'
    catalog['error_model'].pop('session_translation_prior_bound_m',None)
    catalog['error_model']['map_registration_prior']=dict(translation_infinity_bound_m=cfg['se3_translation_bound_m'],rotation_angle_bound_deg=cfg['se3_rotation_bound_deg'])
    catalog['frame_semantics']=dict(history='archive',current='current_local',registration='unknown SE3; not supplied to inference')
    for row in catalog['history']:row['pose_frame']='archive'
    for edge in catalog['edges']:
        a=np.array(catalog['current'][edge['a']]['nominal_T'])[:3,3];b=np.array(catalog['current'][edge['b']]['nominal_T'])[:3,3]
        edge['distance_m']=float(np.linalg.norm(a-b))
    truth.update(session_G=G.tolist(),base_current_nominal_poses=base,scope='unknown currentlocal map coordinateframe; RGBD/physicalscene unchanged by G')
    save(root/'catalog.json',catalog);save(root/'evaluation/truth.json',truth)


def cases(cfg):
    rows=[]
    for family in range(cfg['world_families']):
        repeated=family>=cfg['unique_families']
        for variant in ['identity','se3']:
            rows.append(dict(family=family,variant=variant,history_count=2,repeated=repeated,
                condition=('repeat_' if repeated else 'unique_')+variant+'_h2'))
    for family in range(cfg['unique_families'],cfg['world_families']):
        rows.append(dict(family=family,variant='se3',history_count=1,repeated=True,condition='repeat_se3_h1'))
    return rows


def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    paths=[CFG,CONTRACT,__file__,'inspection_decisions/se3_visibility_v1.py','tests/test_se3_visibility_v1.py',
        'inspection_decisions/projective_visibility_v1.py','inspection_decisions/correspondence_joint_v1.py','inspection_decisions/correspondence_motion_objective_v1.py',
        'scripts/joint_estimation.py','scripts/run_correspondence_joint_pilot_v1.py','scripts/partial_surface_scene.py',
        'comparative_evidence/scene.py','comparative_evidence/interface.py','comparative_evidence/estimation_v1.py',
        'comparative_evidence/evaluation_v1.py','comparative_evidence/evaluation_v3.py','configs/comparative_evidence_interface_v1.json']
    sources={str(Path(p)):digest(p) for p in paths};old=read('results/projective_visibility_v1/frozen_contract.json')
    assert all(digest(p)==sha for p,sha in old['sources'].items())
    save(out/'frozen_contract.json',dict(config=cfg,cases=cases(cfg),sources=sources,status='frozen before fresh generation',old_cores_unchanged=True))
    begin=time.perf_counter()
    for family in range(cfg['world_families']):
        for variant in ['identity','se3']:
            G=np.eye(4) if variant=='identity' else make_G(cfg['seed_start']+family,cfg)
            generate_se3(scene_root(out,family,variant),cfg['seed_start']+family,family>=cfg['unique_families'],G,cfg)
    rows=[]
    for sid,case in enumerate(cases(cfg)):
        root=scene_root(out,case['family'],case['variant']);ep=Episode(root,case['history_count'],2)
        history=[ep.act('retrieve_history',v) for v in cfg['history_views'][:case['history_count']]]
        current=[ep.preview()]+[ep.act('observe_current',v) for v in cfg['current_views'][1:]]
        raw=capture(history,current,ep.metadata()['sensor']);save(out/f'{sid:04d}_observed.json',dict(raw=raw,events=ep.state()['events']))
        print('case',sid,case['condition'],'running',flush=True)
        backend=run_methods(raw,cfg);save(out/f'{sid:04d}_backend.json',backend)
        for method,r in backend['outputs'].items():save(out/f'{sid:04d}_{method}.json',r)
        cost=ep.finish();save(out/f'{sid:04d}_cost.json',cost)
        # ALL reports before truth access. Translation-only controls are intentionally uncorrected.
        truth=read(root/'evaluation/truth.json');trueG=np.array(truth['session_G'])
        for method,result in backend['outputs'].items():
            G=np.array(result['map_transform_current_from_archive']);rotation=float(np.rad2deg(Rotation.from_matrix(G[:3,:3]@trueG[:3,:3].T).magnitude()))
            rows.append(dict(scene=sid,**case,method=method,metric=evaluate(root,result['prediction']),cost=cost,
                unknown=sum(x['label']<0 for x in result['prediction']['assets']),rotation_error_deg=rotation,
                translation_error_m=float(np.linalg.norm(G[:3,3]-trueG[:3,3])),late_selected=backend['late_selected'],
                initial_assignment_hypotheses=backend['registration']['distinct_initial_assignments'],refined_assignment_hypotheses=backend['registration']['distinct_refined_assignments'],retained_poses=len(backend['registration']['retained'])))
        print('case',sid,'complete elapsed',round(time.perf_counter()-begin,1),flush=True)
    summaries={}
    for condition in sorted({r['condition'] for r in rows})+['all']:
        summaries[condition]={}
        for method in cfg['methods']:
            g=[r for r in rows if r['method']==method and (condition=='all' or r['condition']==condition)]
            v={k:sum(r['metric'][k] for r in g) for k in ['changed','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']}
            v.update(unknown=sum(r['unknown'] for r in g),mean_rotation_error_deg=float(np.mean([r['rotation_error_deg'] for r in g])),mean_translation_error_m=float(np.mean([r['translation_error_m'] for r in g])))
            summaries[condition][method]=v
    assert all(digest(p)==sha for p,sha in sources.items())
    save(out/'analysis.json',dict(status='SE3 baseline and latebranch pilot, not innovation confirmation',config=cfg,rows=rows,summaries=summaries,
        world_families=12,renderings=24,RGBD_frames=576,main_acquisition_paths=30,method_outputs=150,exclusions=[]))
    print('PRIMARY',summaries['repeat_se3_h1'],flush=True);print('ALL',summaries['all'],flush=True)


if __name__=='__main__':main()
