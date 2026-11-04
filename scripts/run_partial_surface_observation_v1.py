"""Fixed-evidence, paired-visibility partial surface mechanism pilot."""
import copy,time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.partial_surface_observation_v1 import capture,prepare,solve_directional,apply_surface_gate,surface_seeds
from inspection_decisions.correspondence_joint_v1 import solve as robust_solve
from inspection_decisions.correspondence_motion_objective_v1 import solve as motion_solve
from inspection_decisions.graph_dataset import digest
from scripts.run_correspondence_joint_pilot_v1 import generate
from scripts.run_evidence_value_pilot_v1 import read,save

CFG='configs/partial_surface_observation_v1.json'
CONTRACT='research/partial_surface_observation_contract_v1_2026-10-23.md'


def solver(backends,method,cfg):
    if method in ('surface_directional','surface_isotropic'):
        data=backends['surface'];seeds=surface_seeds(data,cfg)
        if method=='surface_isotropic':
            isotropic=copy.deepcopy(data)
            for f in isotropic['frames']:
                H=np.array(f['H']);H[:]=np.eye(3)/cfg['sigma']**2;f['H']=H.tolist()
            r=solve_directional(isotropic,cfg,seeds)
        else:r=solve_directional(data,cfg,seeds)
        return apply_surface_gate(r,data,cfg)
    if method=='legacy_robust':return robust_solve(backends['legacy'],'robust_multistart',cfg)
    return motion_solve(backends['trimmed' if method=='trimmed_shared' else 'legacy'],'shared_motion',cfg)


def cases(cfg):
    out=[]
    for family in range(cfg['paired_families']):
        for condition,regime in [('clear','repeat_unanchored'),('occluded','repeat_occluded')]:
            out.append(dict(family=family,seed=cfg['seed_start']+family,condition=condition,regime=regime,changed=2 if family%2 else 0))
    for i in range(cfg['normal_control_families']):out.append(dict(family=cfg['paired_families']+i,seed=cfg['seed_start']+100+i,condition='normal',regime='normal',changed=2 if i%2 else 0))
    return out


def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    paths=[CFG,CONTRACT,__file__,'inspection_decisions/partial_surface_observation_v1.py','tests/test_partial_surface_observation_v1.py',
        'inspection_decisions/correspondence_joint_v1.py','inspection_decisions/correspondence_motion_objective_v1.py',
        'scripts/run_correspondence_joint_pilot_v1.py','comparative_evidence/estimation_v1.py','comparative_evidence/interface.py',
        'comparative_evidence/scene.py','comparative_evidence/evaluation_v1.py','comparative_evidence/evaluation_v3.py',
        'configs/comparative_evidence_interface_v1.json']
    sources={str(Path(p)):digest(p) for p in paths}
    old=read('results/correspondence_motion_objective_v1/frozen_contract.json')
    assert all(digest(p)==sha for p,sha in old['sources'].items())
    save(out/'frozen_contract.json',dict(config=cfg,cases=cases(cfg),sources=sources,status='frozen before paired fresh scene generation',old_core_unchanged=True))
    rows=[];begin=time.perf_counter()
    for sid,case in enumerate(cases(cfg)):
        root=out/'scenes'/f'{sid:04d}';generate(root,case['seed'],case['regime'],case['changed'],cfg)
        ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
        raw=capture(h,q);save(out/f'{sid:04d}_observed.json',dict(raw=raw,events=ep.state()['events']))
        backends=prepare(raw,cfg);save(out/f'{sid:04d}_backends.json',backends)
        outputs={}
        for method in cfg['methods']:
            outputs[method]=solver(backends,method,cfg);save(out/f'{sid:04d}_{method}.json',outputs[method])
        cost=ep.finish();save(out/f'{sid:04d}_cost.json',cost)
        # All raw evidence, backends and every method report persist before truth access.
        truth=read(root/'evaluation/truth.json')
        for method,r in outputs.items():
            rows.append(dict(scene=sid,**case,method=method,metric=evaluate(root,r['prediction']),cost=cost,
                unknown=sum(a['label']<0 for a in r['prediction']['assets']),branches=len(r['branches']),distinct=r['distinct_assignment_hypotheses'],
                pose_error_m=float(np.linalg.norm(np.array(r['best']['translation'])-truth['injected_session_translation']))))
        print('scene',sid,'family',case['family'],case['condition'],'elapsed',round(time.perf_counter()-begin,1),flush=True)
    summaries={}
    for condition in ['clear','occluded','normal','all']:
        summaries[condition]={}
        for method in cfg['methods']:
            g=[r for r in rows if r['method']==method and (condition=='all' or r['condition']==condition)]
            v={k:sum(r['metric'][k] for r in g) for k in ['changed','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']}
            v.update(unknown=sum(r['unknown'] for r in g),mean_pose_error_m=float(np.mean([r['pose_error_m'] for r in g])),
                mean_retained_branches=float(np.mean([r['branches'] for r in g])))
            summaries[condition][method]=v
    assert all(digest(p)==sha for p,sha in sources.items())
    save(out/'analysis.json',dict(status='paired visibility model mechanism pilot only',config=cfg,rows=rows,summaries=summaries,
        world_families=cfg['paired_families']+cfg['normal_control_families'],paired_world_families=cfg['paired_families'],
        actual_acquisition_paths=len(cases(cfg)),RGBD_frames=24*len(cases(cfg)),method_outputs=len(rows),exclusions=[]))
    print('OCCLUDED',summaries['occluded'],flush=True);print('ALL',summaries['all'],flush=True)


if __name__=='__main__':main()
