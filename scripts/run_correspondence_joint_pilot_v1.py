"""Real RGB-D ambiguity challenge and truth-separated fixed-evidence pipeline."""
import argparse,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence import scene
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.correspondence_joint_v1 import prepare,solve
from inspection_decisions.graph_dataset import digest
from scripts.run_evidence_value_pilot_v1 import read,save
CFG='configs/correspondence_joint_pilot_v1.json'

def generate(root,seed,regime,changed,cfg):
    sensor=read('configs/comparative_evidence_interface_v1.json');sensor['factors']={}
    original=scene.make_world
    def world(seed,condition):
        past,current,groups,occ,truth=original(seed,condition)
        if regime!='normal':
            centres=np.array(truth['centres']);template=scene.vertices('hex_prism');scale=np.random.default_rng(seed+9).uniform(.70,.85,3)
            history=[[v*scale+c for v in template] for c in centres];present=[[v+np.array(d) for v in parts] for parts,d in zip(history,truth['delta'])]
            groups=np.zeros(8,int)
            if regime=='repeat_anchor':groups[0]=7
            past=[scene.halfspaces(p) for p in history];current=[scene.halfspaces(p) for p in present]
            truth.update(historical_vertices=[[v.tolist() for v in p] for p in history],current_vertices=[[v.tolist() for v in p] for p in present],groups=groups.tolist())
        return past,current,groups,occ if regime=='repeat_occluded' else [],truth
    condition=dict(limited_history=False,coherent_error=True,repeated_appearance=regime!='normal',changed_assets=changed)
    with patch.object(scene,'make_world',world):catalog=scene.generate_scene(root,seed,condition,sensor)
    shift=np.array([(.9 if seed%2 else -.9) if regime!='normal' else .04,0,0])
    for row in catalog['current']:
        token=row['token'];file=root/'payloads'/f'{token}.npz'
        with np.load(file) as z:rgb=z['rgb'].copy();depth=z['depth'].copy();T=z['nominal_T'].copy()
        T[:3,3]+=shift;np.savez_compressed(file,rgb=rgb,depth=depth,nominal_T=T);row['nominal_T']=T.tolist()
    catalog['error_model']['session_translation_prior_bound_m']=cfg['public_session_translation_bound_m']
    # Recheck all nominal path chords after adding the session drift.
    truth=read(root/'evaluation/truth.json');target=np.array([0,0,5.5]);radius=truth['path_enclosing_radius_m']
    for edge in catalog['edges']:
        a=np.array(catalog['current'][edge['a']]['nominal_T'])[:3,3];b=np.array(catalog['current'][edge['b']]['nominal_T'])[:3,3];d=b-a
        u=np.clip((target-a)@d/(d@d),0,1);assert np.linalg.norm(a+u*d-target)>radius
        edge['distance_m']=float(np.linalg.norm(d))
    save(root/'catalog.json',catalog);truth['injected_session_translation']=shift.tolist();truth['regime']=regime;save(root/'evaluation/truth.json',truth)

def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    sources={str(p):digest(p) for p in [Path(CFG),Path(__file__),Path('inspection_decisions/correspondence_joint_v1.py'),Path('research/correspondence_joint_contract_v1_2026-10-23.md'),Path('comparative_evidence/scene.py'),Path('comparative_evidence/interface.py')]}
    save(out/'frozen_contract.json',dict(config=cfg,sources=sources,status='frozen before fresh pilot generation; no tuning'))
    rows=[];begin=time.perf_counter();checks=0
    for sid in range(cfg['scenes']):
        regime=cfg['regimes'][sid//4];seed=cfg['seed_start']+sid;root=out/'scenes'/f'{sid:04d}';changed=2 if sid%2 else 0
        generate(root,seed,regime,changed,cfg);ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
        evidence=prepare(h,q);save(out/f'{sid:04d}_observed.json',dict(evidence=evidence,events=ep.state()['events']))
        outputs={}
        for method in cfg['methods']:
            outputs[method]=solve(evidence,method,cfg);save(out/f'{sid:04d}_{method}.json',outputs[method])
        cost=ep.finish();truth=read(root/'evaluation/truth.json')
        # All observed inputs and all solver reports persisted before first truth evaluation.
        for method,result in outputs.items():
            metric=evaluate(root,result['prediction']);pose_error=float(np.linalg.norm(np.array(result['best']['translation'])-truth['injected_session_translation']))
            rows.append(dict(scene=sid,regime=regime,method=method,metric=metric,cost=cost,pose_error_m=pose_error,
                unknown=sum(a['label']<0 for a in result['prediction']['assets']),branches=len(result['branches']),distinct=result['distinct_assignment_hypotheses']))
            assert result['prediction']['references']==evidence['references'];checks+=1
        print('scene',sid,regime,'elapsed',round(time.perf_counter()-begin,1),flush=True)
    summaries={}
    for regime in cfg['regimes']+['all']:
        summaries[regime]={}
        for method in cfg['methods']:
            g=[r for r in rows if r['method']==method and (regime=='all' or r['regime']==regime)]
            v={k:sum(r['metric'][k] for r in g) for k in ['changed','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']}
            v.update(unknown=sum(r['unknown'] for r in g),mean_pose_error_m=float(np.mean([r['pose_error_m'] for r in g])),mean_distinct_branches=float(np.mean([r['distinct'] for r in g])),mean_retained_branches=float(np.mean([r['branches'] for r in g])))
            summaries[regime][method]=v
    for p,sha in sources.items():assert digest(p)==sha;checks+=1
    save(out/'analysis.json',dict(status='mechanism pilot only, not independent confirmation',rows=rows,summaries=summaries,checks=checks,scene_families=16,method_outputs=64,pair_geometry_preparations=16,association_solver_runs=64,exclusions=[]))
    print(summaries['all'],flush=True)

if __name__=='__main__':main()
