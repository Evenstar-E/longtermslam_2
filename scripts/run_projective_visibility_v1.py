"""Truth-separated fixed-association projective backend comparison."""
import time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture,prepare
from inspection_decisions.correspondence_motion_objective_v1 import solve as motion_solve
from inspection_decisions.correspondence_joint_v1 import solve as robust_solve
from inspection_decisions.graph_dataset import digest
from scripts.run_partial_surface_observation_v1 import cases
from scripts.run_correspondence_joint_pilot_v1 import generate
from scripts.run_evidence_value_pilot_v1 import read,save

CFG='configs/projective_visibility_v1.json'
CONTRACT='research/projective_visibility_contract_v1_2026-10-23.md'


def solver(backend,method,cfg):
    if method=='legacy_shared':return backend['initial']
    if method=='legacy_robust':return robust_solve(backend['legacy'],'robust_multistart',cfg)
    return motion_solve(backend['tables'][method],'shared_motion',cfg)


def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    paths=[CFG,CONTRACT,__file__,'inspection_decisions/projective_visibility_v1.py','tests/test_projective_visibility_v1.py',
        'inspection_decisions/correspondence_joint_v1.py','inspection_decisions/correspondence_motion_objective_v1.py',
        'scripts/run_partial_surface_observation_v1.py','scripts/run_correspondence_joint_pilot_v1.py',
        'comparative_evidence/estimation_v1.py','comparative_evidence/interface.py','comparative_evidence/scene.py',
        'comparative_evidence/evaluation_v1.py','comparative_evidence/evaluation_v3.py','configs/comparative_evidence_interface_v1.json']
    sources={str(Path(p)):digest(p) for p in paths}
    for old in ['results/partial_surface_observation_v1/frozen_contract.json','results/correspondence_motion_objective_v1/frozen_contract.json']:
        assert all(digest(p)==sha for p,sha in read(old)['sources'].items())
    save(out/'frozen_contract.json',dict(config=cfg,cases=cases(cfg),sources=sources,status='frozen before generation',old_cores_unchanged=True))
    rows=[];begin=time.perf_counter()
    for sid,case in enumerate(cases(cfg)):
        root=out/'scenes'/f'{sid:04d}';generate(root,case['seed'],case['regime'],case['changed'],cfg)
        ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
        raw=capture(h,q,ep.metadata()['sensor']);save(out/f'{sid:04d}_observed.json',dict(raw=raw,events=ep.state()['events']))
        print('scene',sid,case['condition'],'observed; fitting projective backends',flush=True)
        backend=prepare(raw,cfg);save(out/f'{sid:04d}_backend.json',backend);outputs={}
        for method in cfg['methods']:outputs[method]=solver(backend,method,cfg);save(out/f'{sid:04d}_{method}.json',outputs[method])
        cost=ep.finish();save(out/f'{sid:04d}_cost.json',cost)
        # Raw inputs, all fits and ALL reports persist before evaluator truth access.
        truth=read(root/'evaluation/truth.json')
        for method,r in outputs.items():rows.append(dict(scene=sid,**case,method=method,metric=evaluate(root,r['prediction']),cost=cost,
            unknown=sum(a['label']<0 for a in r['prediction']['assets']),branches=len(r['branches']),
            pose_error_m=float(np.linalg.norm(np.array(r['best']['translation'])-truth['injected_session_translation']))))
        print('scene',sid,'complete elapsed',round(time.perf_counter()-begin,1),flush=True)
    summaries={}
    for condition in ['clear','occluded','normal','all']:
        summaries[condition]={}
        for method in cfg['methods']:
            g=[r for r in rows if r['method']==method and (condition=='all' or r['condition']==condition)]
            v={k:sum(r['metric'][k] for r in g) for k in ['changed','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']}
            v.update(unknown=sum(r['unknown'] for r in g),mean_pose_error_m=float(np.mean([r['pose_error_m'] for r in g])))
            summaries[condition][method]=v
    assert all(digest(p)==sha for p,sha in sources.items())
    save(out/'analysis.json',dict(status='two-stage projective backend pilot; not joint identity/SE3/confirmation',config=cfg,
        rows=rows,summaries=summaries,world_families=12,paired_world_families=8,actual_main_acquisition_paths=20,RGBD_frames=480,method_outputs=len(rows),exclusions=[]))
    print('OCCLUDED',summaries['occluded'],flush=True);print('ALL',summaries['all'],flush=True)


if __name__=='__main__':main()
