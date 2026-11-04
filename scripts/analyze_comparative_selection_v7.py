"""Audit full policies and complete common-state action branches; DEV only."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import spearmanr
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v7 import run_selection_v7


def aggregate(rows):
    keys=('qualified_tp','tp','fp','bad_motion_reports','known','changed','assets','task_loss','identity_wrong_references')
    result={k:sum(r[k] for r in rows) for k in keys}
    result['mean_travel_m']=float(np.mean([r['cost']['travel_m'] for r in rows]))
    return result


def main():
    base=Path('results/comparative_selection_v7');branchbase=Path('results/comparative_event_branches_v7')
    cfg=json.loads(Path('configs/comparative_selection_v7.json').read_text())
    rows=[];saved_cache={};branches=[];checks=0;manifests=[]
    for directory in (base,branchbase):
        for shard in range(2):
            root=directory/f'shard_{shard}';m=json.loads((root/'manifest.json').read_text());manifests.append(m)
            for p,d in m['sources'].items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d;checks+=1
            assert (root/'frozen_config.json').read_bytes()==Path('configs/comparative_selection_v7.json').read_bytes();checks+=1
            assert not m['exclusions'];checks+=1
            if directory==base:
                assert m['executions']==96 and m['evaluations']==288;checks+=1
                for row in map(json.loads,(root/'metrics.jsonl').read_text().splitlines()):
                    sid,method=row['scene'],row['method'];key=sid,method
                    if key not in saved_cache:
                        target=root/'predictions'/f'{sid:04d}_{method}.json';raw=target.read_bytes()
                        assert hashlib.sha256(raw).hexdigest()==row['prediction_sha256'];checks+=1
                        saved=json.loads(raw);saved_cache[key]=saved
                        assert saved['cost']['history_views']==4 and saved['cost']['additional_current_views']==2 and saved['cost']['returned'];checks+=1
                        known={'history':{0},'current':{0}}
                        for d in saved['prediction']['decisions']:
                            assert d['score']==max(a['score'] for a in d['alternatives']);checks+=1
                            for a in d['alternatives']:
                                assert a['view'] not in known[a['epoch']];checks+=1
                                tokens={f'{epoch}_{v:02d}' for epoch in known for v in known[epoch]}
                                assert set(a['source_tokens'])<=tokens;checks+=1
                                assert set(a['gray_sources'])<={f"{a['epoch']}_{v:02d}" for v in known[a['epoch']]};checks+=1
                                assert .01-1e-12<=a['mean_detection_probability']<=.99+1e-12;checks+=1
                                assert a['informative_events']==(not method.startswith('constant_'));checks+=1
                            known[d['epoch']].add(d['view'])
                        assert len(saved['prediction']['decisions'])==5;checks+=1
                    p=saved_cache[key]['prediction']
                    prediction=p if row['variant']=='risk' else p['terminal_alternatives'][row['variant']]
                    score=evaluate(Path(cfg['source_data'])/f'{sid:04d}',prediction)
                    assert all(row[k]==v for k,v in score.items());checks+=1
                    rows.append(row)
            else:
                assert m['scenes']==24 and m['branches']==528;checks+=1
                for row in map(json.loads,(root/'metrics.jsonl').read_text().splitlines()):
                    sid,epoch,view=row['scene'],row['epoch'],row['view']
                    target=root/'branches'/f'{sid:04d}_{epoch}_{view:02d}.json';raw=target.read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==row['prediction_sha256'];checks+=1
                    saved=json.loads(raw);scene=Path(cfg['source_data'])/f'{sid:04d}'
                    for key,p in (('initial',saved['initial']),('initial_point',saved['initial_point']),
                                  ('after',saved['prediction']),('after_point',saved['point_prediction'])):
                        assert evaluate(scene,p)==row[key];checks+=1
                    assert saved['state']['closed'] and saved['cost']['returned'];checks+=1
                    assert saved['state']['history_read']==([0,view] if epoch=='history' else [0]);checks+=1
                    assert saved['state']['current_observed']==([0,view] if epoch=='current' else [0]);checks+=1
                    assert set(row['scores']['audit']['source_tokens'])<= {'history_00','current_00'};checks+=1
                    assert row['scores']==saved['scores'] and row['diagnostics']==saved['diagnostics'];checks+=1
                    branches.append(row)
    assert len(rows)==576 and len(branches)==1056;checks+=1
    # All old controls retained with scene rows, not counted as new runs.
    for directory in ('results/comparative_selection_v5','results/comparative_selection_v6'):
        for shard in range(2):
            d=Path(directory)/f'shard_{shard}';m=json.loads((d/'manifest.json').read_text())
            for p,h in m['sources'].items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;checks+=1
            old=list(map(json.loads,(d/'metrics.jsonl').read_text().splitlines()))
            for row in old:
                if directory.endswith('v6') and row['method'] not in ('uniform_candidate','uniform_risk'):continue
                name=row['method']
                if name=='support_candidate':name='weighted_candidate'
                elif name=='support_risk':name='weighted_risk'
                row['method']=name;rows.append(row)
    data=json.loads(Path('results/comparative_interface_v1/manifest.json').read_text());cells=defaultdict(list)
    for s in data['scenes']:
        root=Path(cfg['source_data'])/f"{s['scene']:04d}"
        for p,h in s['hashes'].items():
            assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h;checks+=1
        cells[json.dumps(s['condition'],sort_keys=True)].append(s['scene'])
    assert len(cells)==16 and all(len(c)==3 for c in cells.values());checks+=1
    methods=['legacy_candidate','legacy_risk','legacy_schur','weighted_candidate','weighted_risk',
        'uniform_candidate','uniform_risk','support_coverage','represented_coverage']+cfg['methods']
    groups={(m,v):sorted([r for r in rows if r['method']==m and r['variant']==v],key=lambda r:r['scene'])
            for m in methods for v in ('risk','point','two_sigma')}
    assert all(len(g)==48 for g in groups.values());checks+=1
    summaries={m:{v:aggregate(groups[m,v]) for v in ('risk','point','two_sigma')} for m in methods}
    rng=np.random.default_rng(cfg['analysis']['seed']);draws=np.stack([
        np.concatenate([np.array(c)[rng.integers(0,3,3)] for c in cells.values()]) for _ in range(cfg['analysis']['resamples'])])
    comparisons={}
    for left,right in cfg['analysis']['comparisons']:
        a,b=groups[left,'risk'],groups[right,'risk'];delta=np.array([x['qualified_tp']-y['qualified_tp'] for x,y in zip(a,b)])
        denominator=np.array([r['changed'] for r in a]);dist=delta[draws].sum(1)/denominator[draws].sum(1)*100
        comparisons[left+' minus '+right]=dict(qualified_difference=int(delta.sum()),difference_pp=float(delta.sum()/72*100),
            exploratory_95_ci_pp=np.quantile(dist,[.025,.975]).tolist(),scene_differences=delta.tolist())
    branch_summary,branch_checks=analyze_branches(branches,saved_cache,cfg)
    checks+=branch_checks
    replay=execute_policy(Path(cfg['source_data'])/'0001',*cfg['budget'],lambda port:run_selection_v7(port,'events_risk'))
    assert replay['prediction']==saved_cache[1,'events_risk']['prediction'];checks+=1
    result=dict(scope='48 reused DEV bases;192 new full policies,1056 offline single-action branches; no independent confirmation',
        summaries=summaries,comparisons=comparisons,branch_validation=branch_summary,checks=checks,exclusions=[],
        seconds=[m['seconds'] for m in manifests],multiplicity='6 exploratory paired full-task contrasts, no familywise confirmation',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for m in methods:print(m,json.dumps(summaries[m]['risk']))
    print('comparisons',json.dumps(comparisons));print('branches',json.dumps(branch_summary));print('checks',checks)


def analyze_branches(branches,policies,cfg):
    checks=0;scene_records=[]
    methods=cfg['methods']+['weighted_candidate','uniform_risk','support_coverage']
    for sid in range(48):
        rows=sorted([r for r in branches if r['scene']==sid],key=lambda r:(r['epoch']!='history',r['view']))
        assert len(rows)==22 and len({(r['epoch'],r['view']) for r in rows})==22;checks+=1
        initial=rows[0]['initial'];assert all(r['initial']==initial for r in rows);checks+=1
        improvement=np.array([initial['task_loss']-r['after']['task_loss'] for r in rows])
        for method in methods:
            score=np.array([r['scores']['normalized'][method] for r in rows]);j=int(score.argmax());chosen=rows[j]
            rho=None if np.ptp(score)<1e-12 or np.ptp(improvement)<1e-12 else float(spearmanr(score,improvement).statistic)
            if policies is not None and method in cfg['methods']:
                d=policies[sid,method]['prediction']['decisions'][0]
                assert (chosen['epoch'],chosen['view'])==(d['epoch'],d['view']);checks+=1
                assert chosen['scores']['normalized'][method]==d['score'];checks+=1
            scene_records.append(dict(scene=sid,method=method,action=[chosen['epoch'],chosen['view']],
                rank_correlation=rho,initial_taskloss=initial['task_loss'],after_taskloss=chosen['after']['task_loss'],
                realized_loss_drop=float(improvement[j]),hindsight_single_realization_gap=float(improvement.max()-improvement[j]),
                qualified_after=chosen['after']['qualified_tp'],fp_after=chosen['after']['fp'],
                event_spread=chosen['scores']['audit']['particle_detection_spread']))
    summaries={}
    for method in methods:
        r=[s for s in scene_records if s['method']==method];rho=[s['rank_correlation'] for s in r if s['rank_correlation'] is not None]
        summaries[method]=dict(qualified_after_one_action=sum(s['qualified_after'] for s in r),
            fp_after_one_action=sum(s['fp_after'] for s in r),total_realized_loss_drop=sum(s['realized_loss_drop'] for s in r),
            total_taskloss_after_one_action=sum(s['after_taskloss'] for s in r),
            mean_hindsight_single_realization_gap=float(np.mean([s['hindsight_single_realization_gap'] for s in r])),
            mean_within_scene_rank_rho=float(np.mean(rho)) if rho else None,nonconstant_rank_scenes=len(rho),
            mean_selected_particle_event_spread=float(np.mean([s['event_spread'] for s in r])))
    calibration=dict(branches=len(branches),asset_events=len(branches)*8,
        observed_detection_fraction=float(np.mean([np.mean(r['observed_mask']) for r in branches])),
        predicted_detection_fraction=float(np.mean([np.mean(r['scores']['pi']) for r in branches])),
        mean_brier=float(np.mean([r['diagnostics']['event_brier'] for r in branches])),
        mean_event_only_nll=float(np.mean([r['diagnostics']['event_only_nll'] for r in branches])),
        mean_full_feature_nll=float(np.mean([r['diagnostics']['full_feature_nll'] for r in branches])),
        mean_constant_event_feature_nll=float(np.mean([r['diagnostics']['constant_event_feature_nll'] for r in branches])))
    probability=np.array([r['scores']['pi'] for r in branches]).ravel()
    observed=np.array([r['observed_mask'] for r in branches],float).ravel()
    calibration['brier_always_detected_baseline']=float(np.mean((1-observed)**2))
    calibration['brier_fixed_099_detection_baseline']=float(np.mean((.99-observed)**2))
    bins=[]
    for low,high in ((0.,.1),(.1,.5),(.5,.9),(.9,1.)):
        use=(probability>=low)&(probability<high)
        bins.append(dict(interval=[low,high],asset_event_count=int(use.sum()),
            mean_predicted=float(probability[use].mean()) if use.any() else None,
            empirical_detection=float(observed[use].mean()) if use.any() else None))
    calibration['descriptive_probability_bins']=bins
    return dict(scope='All22 legal initial actions, ALL48 common H0/C0 states. Actual branch outcomes held out of scoring. Not multi-step optimality or all later states.',
        top_action_summaries=summaries,calibration=calibration,scene_records=scene_records,
        limits='Oracle gap uses one realized outcome per action and is hindsight only; descriptive rho has no independent-action pvalue; Brier same marginalpi for event/constant control'),checks


if __name__=='__main__':main()
