"""Integrity, scene-paired effects and fixed-observable branch validation."""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.special import logsumexp
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v8 import run_selection_v8
from comparative_evidence.interface import Episode, shortest_path
from comparative_evidence.geometry_modes_v2 import estimate_modes, mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.event_likelihood_v7 import held_clouds
from comparative_evidence.ray_likelihood_v8 import forecast, encode_depth, log_likelihood

CHECKS=0


def check(value):
    global CHECKS
    assert value
    CHECKS+=1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readrows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def aggregate(rows):
    keys=('qualified_tp','tp','fp','bad_motion_reports','known','changed','assets','task_loss','identity_wrong_references')
    return dict(**{k:sum(r[k] for r in rows) for k in keys},
                mean_travel_m=float(np.mean([r['cost']['travel_m'] for r in rows])),
                mean_policy_seconds=float(np.mean([r['timings']['policy_wall_excluding_api_s'] for r in rows])))


def main():
    cfg=json.loads(Path('configs/comparative_selection_v8.json').read_text())
    base=Path('results/comparative_selection_v8'); bb=Path('results/comparative_ray_branches_v8')
    rows=[]; branches=[]; policies={}; manifests=[]
    for directory in (base,bb):
        for shard in range(2):
            root=directory/f'shard_{shard}';m=json.loads((root/'manifest.json').read_text());manifests.append(m)
            check(not m['exclusions']);check(m['scenes']==24)
            for category in ('sources','data'):
                for path,digest in m[category].items():check(sha(path)==digest)
            check((root/'frozen_config.json').read_bytes()==Path('configs/comparative_selection_v8.json').read_bytes())
            if directory==base:
                check(m['executions']==96 and m['evaluations']==288)
                for row in readrows(root/'metrics.jsonl'):
                    sid,method=row['scene'],row['method'];key=(sid,method)
                    if key not in policies:
                        pfile=root/'predictions'/f'{sid:04d}_{method}.json'
                        check(sha(pfile)==row['prediction_sha256'])
                        saved=json.loads(pfile.read_text());policies[key]=saved
                        check(saved['cost']['history_views']==4 and saved['cost']['additional_current_views']==2 and saved['cost']['returned'])
                        known={'history':{0},'current':{0}}
                        for d in saved['prediction']['decisions']:
                            check(d['score']==max(a['score'] for a in d['alternatives']))
                            check((d['epoch'],d['view'])==next((a['epoch'],a['view']) for a in d['alternatives'] if a['score']==d['score']))
                            tokens={f'{e}_{v:02d}' for e in known for v in known[e]}
                            for a in d['alternatives']:
                                check(a['view'] not in known[a['epoch']]);check(set(a['source_tokens'])<=tokens)
                                check(set(a['gray_sources'])<={f"{a['epoch']}_{v:02d}" for v in known[a['epoch']]})
                                check(len(set(a['pixels']))==a['rays'] and 0<a['rays']<=24)
                                check(all(0<=v<128*96 for v in a['pixels']));check(0<=a['represented_hit_fraction']<=1)
                            known[d['epoch']].add(d['view'])
                        check(len(saved['prediction']['decisions'])==5)
                    p=policies[key]['prediction'];prediction=p if row['variant']=='risk' else p['terminal_alternatives'][row['variant']]
                    metric=evaluate(Path(cfg['source_data'])/f'{sid:04d}',prediction)
                    check(all(row[k]==v for k,v in metric.items()));rows.append(row)
            else:
                check(m['branches']==528)
                for row in readrows(root/'metrics.jsonl'):
                    sid=row['scene'];frozen=root/'branches'/f'{sid:04d}_frozen_scores.json'
                    check(sha(frozen)==row['frozen_scores_sha256'])
                    frozen_rows=json.loads(frozen.read_text()); fr=next(r for r in frozen_rows if (r['epoch'],r['view'])==(row['epoch'],row['view']))
                    check({k:v for k,v in fr.items() if k not in ('epoch','view')}==row['scores'])
                    check(sha(row['cached_source'])==row['cached_sha256'])
                    old=json.loads(Path(row['cached_source']).read_text());scene=Path(cfg['source_data'])/f'{sid:04d}'
                    check(evaluate(scene,old['prediction'])==row['after']);check(evaluate(scene,old['initial'])==row['initial'])
                    check(set(row['scores']['audit']['source_tokens'])<={'history_00','current_00'})
                    check(len(row['observed_symbols'])==row['scores']['audit']['rays']);branches.append(row)
    check(len(rows)==576 and len(branches)==1056 and len(policies)==192)
    # Import all13 old controls, preserved under their own names. New policies
    # are4, not17; reuse is explicit and not added to execution counts.
    oldsummary=json.loads(Path('results/comparative_selection_v7/analysis.json').read_text())['summaries']
    for version in ('v5','v6','v7'):
        for shard in range(2):
            oldroot=Path(f'results/comparative_selection_{version}')/f'shard_{shard}'
            oldm=json.loads((oldroot/'manifest.json').read_text())
            for path,digest in oldm['sources'].items():check(sha(path)==digest)
    controls=[]
    for version,wanted in (('v5',{'support_coverage'}),('v6',{'uniform_risk'})):
        for shard in range(2):
            controls += [r for r in readrows(Path(f'results/comparative_selection_{version}')/f'shard_{shard}'/'metrics.jsonl') if r['method'] in wanted]
    grouped={(m,v):sorted([r for r in rows+controls if r['method']==m and r['variant']==v],key=lambda r:r['scene'])
             for m in cfg['methods']+['uniform_risk','support_coverage'] for v in ('risk','point','two_sigma')}
    check(all(len(r)==48 for r in grouped.values()))
    summaries=dict(oldsummary)
    summaries.update({m:{v:aggregate(grouped[m,v]) for v in ('risk','point','two_sigma')} for m in cfg['methods']})
    data=json.loads(Path('results/comparative_interface_v1/manifest.json').read_text());cells=defaultdict(list)
    for s in data['scenes']:cells[json.dumps(s['condition'],sort_keys=True)].append(s['scene'])
    check(len(cells)==16 and all(len(c)==3 for c in cells.values()))
    rng=np.random.default_rng(cfg['bootstrap']['seed'])
    draws=np.stack([np.concatenate([np.array(c)[rng.integers(0,3,3)] for c in cells.values()]) for _ in range(3000)])
    comparisons={}
    for left,right in cfg['prespecified_pairs']:
        a,b=grouped[left,'risk'],grouped[right,'risk']
        delta=np.array([x['qualified_tp']-y['qualified_tp'] for x,y in zip(a,b)])
        neg=np.array([x['assets']-x['changed'] for x in a]);positive=np.array([x['changed'] for x in a])
        fpdelta=np.array([x['fp']-y['fp'] for x,y in zip(a,b)])
        dist=delta[draws].sum(1)/positive[draws].sum(1)*100
        fpd=fpdelta[draws].sum(1)/neg[draws].sum(1)*100
        travel_a=np.array([r['cost']['travel_m'] for r in a]);travel_b=np.array([r['cost']['travel_m'] for r in b])
        ratios=travel_a[draws].sum(1)/travel_b[draws].sum(1)
        ci=np.quantile(dist,[.025,.975]);fpci=np.quantile(fpd,[.025,.975]);ratio_ci=np.quantile(ratios,[.025,.975])
        comparisons[left+' minus '+right]=dict(difference_pp=float(delta.sum()/72*100),exploratory_95_ci_pp=ci.tolist(),
            scene_differences=delta.tolist(),fp_difference_pp=float(fpdelta.sum()/312*100),fp_95_ci_pp=fpci.tolist(),
            travel_ratio=float(travel_a.sum()/travel_b.sum()),travel_ratio_95_ci=ratio_ci.tolist(),
            performance_gate=bool(delta.sum()/72>=.05 and ci[0]>0 and fpci[1]<=1),
            travel_only_efficiency_gate=bool(ci[0]>=-2 and fpci[1]<=1 and ratio_ci[1]<=.8))
    minimum_travel=[]
    for sid in range(48):
        meta=json.loads((Path(cfg['source_data'])/f'{sid:04d}'/'catalog.json').read_text())
        d=np.array([[shortest_path(meta['edges'],i,j)[0] for j in range(12)] for i in range(12)])
        minimum_travel.append(min(d[0,i]+d[i,j]+d[j,0] for i in range(1,12) for j in range(i+1,12)))
        for method in cfg['methods']:check(policies[sid,method]['cost']['travel_m']+1e-9>=minimum_travel[-1])
    strong=grouped['uniform_risk','risk'];strongtravel=np.array([r['cost']['travel_m'] for r in strong])
    minimum_travel=np.array(minimum_travel)
    gate_feasibility=dict(scope='Same mandatory2distinct current views plus return; metadata shortest-path lower bound, not executable evidence utility.',
        maximum_completion_gain_vs_strong_pp=(72-sum(r['qualified_tp'] for r in strong))/72*100,
        minimum_mean_travel_m=float(minimum_travel.mean()),minimum_possible_travel_ratio_vs_strong=float(minimum_travel.sum()/strongtravel.sum()),
        minimum_ratio_scene_bootstrap_95_ci=np.quantile(minimum_travel[draws].sum(1)/strongtravel[draws].sum(1),[.025,.975]).tolist(),
        limitation='Travel only; does not prove impossibility for bytes/CPU/composite cost or variable-count early stopping. Frozen old gates retained.')
    scene_records=[]
    for sid in range(48):
        br=[r for r in branches if r['scene']==sid];check(len(br)==22)
        check(len({(r['epoch'],r['view']) for r in br})==22)
        improvement=np.array([r['initial']['task_loss']-r['after']['task_loss'] for r in br])
        for method in cfg['methods']:
            scores=np.array([r['scores']['normalized'][method] for r in br]);j=int(scores.argmax());r=br[j]
            d=policies[sid,method]['prediction']['decisions'][0]
            check((r['epoch'],r['view'])==(d['epoch'],d['view']));check(scores[j]==d['score'])
            rho=float(spearmanr(scores,improvement).statistic) if np.ptp(scores)>1e-12 and np.ptp(improvement)>1e-12 else None
            scene_records.append(dict(scene=sid,method=method,action=[r['epoch'],r['view']],qualified=r['after']['qualified_tp'],
                fp=r['after']['fp'],task_loss=r['after']['task_loss'],rho=rho,hindsight_gap=float(improvement.max()-improvement[j])))
    branchsummary={}
    for method in cfg['methods']:
        ss=[r for r in scene_records if r['method']==method];rho=[r['rho'] for r in ss if r['rho'] is not None]
        branchsummary[method]=dict(qualified=sum(r['qualified'] for r in ss),fp=sum(r['fp'] for r in ss),task_loss=sum(r['task_loss'] for r in ss),
            mean_rho=float(np.mean(rho)) if rho else None,nonconstant_scenes=len(rho),mean_hindsight_gap=float(np.mean([r['hindsight_gap'] for r in ss])))
    # Replay observation extraction/model law on prespecified opposite-view
    # counterexample0H6, without evaluation geometry in forecast.
    ep=Episode(Path(cfg['source_data'])/'0000',*cfg['budget']);h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
    fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
    m=forecast(('history',6),fit,meta,held_clouds(fit,h,q,meta),mixture_particles(fit)[0])
    obs=ep.act('retrieve_history',6);value=encode_depth(obs,m['pixels']);ep.finish()
    r=next(r for r in branches if r['scene']==0 and r['epoch']=='history' and r['view']==6)
    check(value.tolist()==r['observed_symbols']);check(m['audit']==r['scores']['audit'])
    for shared in (True,False):check(float(-logsumexp(log_likelihood(m,value,shared))+np.log(32))==r['nll'][str(shared)])
    replay=execute_policy(Path(cfg['source_data'])/'0001',*cfg['budget'],lambda p:run_selection_v8(p,'shared_risk'))
    check(replay['prediction']==policies[1,'shared_risk']['prediction'])
    calibration=dict(branches=1056,ray_observations=sum(len(r['observed_symbols']) for r in branches),
        shared_nll=float(np.mean([r['nll']['True'] for r in branches])),factorized_nll=float(np.mean([r['nll']['False'] for r in branches])),
        uniform_nll=float(np.mean([r['uniform_depth_nll'] for r in branches])),depth_brier=float(np.mean([r['depth_brier'] for r in branches])),
        uniform_depth_brier=1-1/66,scope='Identical depth domain only; not comparable to v7asset-event Brier/NLL; no physical calibration guarantee')
    # Added after looking at branch scores: explicitly exploratory predictive
    # diagnostics, not one of the five prespecified task contrasts.
    nll_scene=np.array([[np.mean([r['nll']['True']-r['nll']['False'] for r in branches if r['scene']==sid]),
                         np.mean([r['nll']['True']-r['uniform_depth_nll'] for r in branches if r['scene']==sid])]
                        for sid in range(48)])
    calibration['posthoc_scene_paired_nll_differences']=dict(shared_minus_factorized_and_uniform=nll_scene.mean(0).tolist(),
        exploratory_95_ci=np.quantile(nll_scene[draws].mean(1),[.025,.975],axis=0).tolist(),
        note='Posthoc scene-stratified descriptive comparison; likelihood gain is not action-value or novelty gain')
    supportpath=bb/'belief_support_diagnostic.json';support=json.loads(supportpath.read_text())
    for path,digest in support['inputs_sha256'].items():check(sha(path)==digest)
    check(sha('scripts/audit_ray_belief_support_v8.py')==support['script_sha256'])
    check(len(support['records'])==8448)
    check(float(np.mean([r['represented_within_task_tolerance'] for r in support['records']]))==support['within_10cm_fraction'])
    supportsummary={k:v for k,v in support.items() if k not in ('records','inputs_sha256','per_scene_fraction')}
    result=dict(scope='48 reusedDEV;192newfullpolicies576terminalevals;1056newrayobservations with cachedv7afterfits,not1056newfits',
        summaries=summaries,comparisons=comparisons,gate_feasibility=gate_feasibility,branch_summary=branchsummary,branch_scene_records=scene_records,
        calibration=calibration,belief_support_posthoc=supportsummary,opposite_view_counterexample=r,checks=CHECKS,exclusions=[],seconds=[m['seconds'] for m in manifests],
        inference='5 exploratory scene-stratified paired comparisons; no multiplicity-adjusted confirmation',
        contribution_contract_sha256=sha('research/ray_v8_contribution_contract_2026-10-22.md'),
        analyzer_sha256=sha(__file__),python=sys.version)
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fields=['method','variant','qualified_tp','tp','fp','bad_motion_reports','known','task_loss','mean_travel_m']
    with (base/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,variants in summaries.items():
            for variant,metric in variants.items():writer.writerow(dict(method=method,variant=variant,**{k:metric[k] for k in fields[2:]}))
    for method in cfg['methods']:print(method,json.dumps(summaries[method]['risk']))
    print('comparisons',json.dumps(comparisons));print('branches',json.dumps(branchsummary));print('calibration',json.dumps(calibration));print('checks',CHECKS)


if __name__=='__main__':main()
