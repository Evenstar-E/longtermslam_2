"""Prospectively frozen, fixed-evidence structural-model pilot."""
import time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.correspondence_joint_v1 import prepare,solve as old_solve
from inspection_decisions.correspondence_motion_objective_v1 import solve as new_solve
from inspection_decisions.graph_dataset import digest
from scripts.run_correspondence_joint_pilot_v1 import generate
from scripts.run_evidence_value_pilot_v1 import read,save

CFG='configs/correspondence_motion_objective_v1.json'
CONTRACT='research/correspondence_motion_objective_contract_v1_2026-10-23.md'


def solver(data,method,cfg):
    return (old_solve if method in ('joint_branches','robust_multistart') else new_solve)(data,method,cfg)


def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    paths=[CFG,CONTRACT,__file__,'inspection_decisions/correspondence_motion_objective_v1.py',
           'inspection_decisions/correspondence_joint_v1.py','scripts/run_correspondence_joint_pilot_v1.py',
           'comparative_evidence/estimation_v1.py','comparative_evidence/scene.py','comparative_evidence/interface.py',
           'comparative_evidence/evaluation_v1.py','comparative_evidence/evaluation_v3.py',
           'configs/comparative_evidence_interface_v1.json','tests/test_correspondence_motion_objective_v1.py']
    sources={str(Path(p)):digest(p) for p in paths}
    old=read('results/correspondence_joint_pilot_v1/frozen_contract.json')
    for p,sha in old['sources'].items():assert digest(p)==sha
    save(out/'frozen_contract.json',dict(config=cfg,sources=sources,status='frozen before generation',old_pilot_core_unchanged=True))
    rows=[];begin=time.perf_counter()
    per_regime=cfg['scenes']//len(cfg['regimes'])
    for sid in range(cfg['scenes']):
        regime=cfg['regimes'][sid//per_regime];root=out/'scenes'/f'{sid:04d}'
        generate(root,cfg['seed_start']+sid,regime,2 if sid%2 else 0,cfg)
        ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)]
        q=[ep.preview(),ep.act('observe_current',1)];data=prepare(h,q)
        save(out/f'{sid:04d}_observed.json',dict(evidence=data,events=ep.state()['events']))
        outputs={}
        for method in cfg['methods']:
            start=time.perf_counter();outputs[method]=solver(data,method,cfg)
            save(out/f'{sid:04d}_{method}.json',outputs[method])
            outputs[method]['solver_elapsed_s']=time.perf_counter()-start
        cost=ep.finish();save(out/f'{sid:04d}_cost.json',cost)
        # No evaluator truth is used by prepare/solve; evaluation begins only after ALL reports persist.
        truth=read(root/'evaluation/truth.json')
        for method,r in outputs.items():
            metric=evaluate(root,r['prediction'])
            rows.append(dict(scene=sid,regime=regime,method=method,metric=metric,cost=cost,
                unknown=sum(a['label']<0 for a in r['prediction']['assets']),branches=len(r['branches']),
                distinct=r['distinct_assignment_hypotheses'],solver_elapsed_s=r['solver_elapsed_s'],
                pose_error_m=float(np.linalg.norm(np.array(r['best']['translation'])-truth['injected_session_translation']))))
        print('scene',sid,regime,'elapsed',round(time.perf_counter()-begin,1),flush=True)
    summaries={}
    for regime in cfg['regimes']+['all']:
        summaries[regime]={}
        for method in cfg['methods']:
            g=[r for r in rows if r['method']==method and (regime=='all' or r['regime']==regime)]
            v={k:sum(r['metric'][k] for r in g) for k in ['changed','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']}
            v.update(unknown=sum(r['unknown'] for r in g),mean_pose_error_m=float(np.mean([r['pose_error_m'] for r in g])),
                mean_retained_branches=float(np.mean([r['branches'] for r in g])),mean_distinct_branches=float(np.mean([r['distinct'] for r in g])),
                mean_solver_elapsed_s=float(np.mean([r['solver_elapsed_s'] for r in g])))
            summaries[regime][method]=v
    for p,sha in sources.items():assert digest(p)==sha
    save(out/'analysis.json',dict(status='bounded model mechanism pilot only',config=cfg,rows=rows,summaries=summaries,
        scene_families=cfg['scenes'],actual_acquisition_paths=cfg['scenes'],method_outputs=len(rows),
        RGBD_frames=24*cfg['scenes'],exclusions=[],initial_freeze_unchanged=True))
    print(summaries['all'],flush=True)


if __name__=='__main__':main()
