"""Independent fixed-evidence and active-policy evaluation, all reused dev data."""
import hashlib,json
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v3 import run_selection_v3


def aggregate(rows):
    fields=('qualified_tp','tp','fp','bad_motion_reports','known','wrong_known','changed','unchanged','assets','task_loss')
    result={k:sum(r[k] for r in rows) for k in fields}
    result.update(completion=result['qualified_tp']/result['changed'],coverage=result['known']/result['assets'])
    ell={k:sum(r['nominal_ellipsoid'][k] for r in rows) for k in rows[0]['nominal_ellipsoid']}
    ell['changed_coverage']=ell['changed_inside']/max(ell['changed_total'],1)
    ell['unchanged_coverage']=ell['unchanged_inside']/max(ell['unchanged_total'],1)
    result['nominal_ellipsoid']=ell
    if 'timings' in rows[0]:
        result['median_policy_compute_s']=float(np.median([r['timings']['policy_wall_excluding_api_s'] for r in rows]))
        result['mean_travel_m']=float(np.mean([r['cost']['travel_m'] for r in rows]))
    return result


def main():
    cfg=json.loads(Path('configs/comparative_selection_v3.json').read_text());source=Path(cfg['source_data'])
    fixed=Path('results/comparative_fixed_v3');active=Path('results/comparative_active_v3')
    frows=[json.loads(s) for s in (fixed/'metrics.jsonl').read_text().splitlines()]
    arows=[json.loads(s) for s in (active/'metrics.jsonl').read_text().splitlines()]
    assert len(frows)==1680 and len(arows)==432
    checks=2
    for root in (fixed,active):
        manifest=json.loads((root/'manifest.json').read_text())
        for p,digest in manifest['sources'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest;checks+=1
    for previous in ('results/comparative_selection_v2','results/comparative_selection_v1','results/comparative_interface_v1'):
        m=json.loads((Path(previous)/'manifest.json').read_text())
        for p,digest in m['source_sha256'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest;checks+=1
    data=json.loads((source/'manifest.json').read_text())
    for scene in data['scenes']:
        root=source/'scenes'/f"{scene['scene']:04d}"
        for p,digest in scene['hashes'].items():
            assert hashlib.sha256((root/p).read_bytes()).hexdigest()==digest;checks+=1
    oldrows={(r['scene'],r['method']):r for r in map(json.loads,Path('results/comparative_selection_v2/metrics.jsonl').read_text().splitlines())}
    saved_cache={};failure_records=[]
    for r in frows:
        sid,method,variant=r['scene'],r['method'],r['variant'];key=sid,method
        if key not in saved_cache:
            path=fixed/'predictions'/f'{sid:04d}_{method}.json';raw=path.read_bytes()
            saved=json.loads(raw);saved_cache[key]=saved
            assert hashlib.sha256(raw).hexdigest()==r['prediction_sha256'];checks+=1
            source_prediction=Path(cfg['source_traces'])/'predictions'/f'{sid:04d}_{method}.json'
            assert hashlib.sha256(source_prediction.read_bytes()).hexdigest()==saved['source_prediction_sha256'];checks+=1
            original=json.loads(source_prediction.read_text())['prediction']
            for a,b in zip(saved['uncertainty'],original['mode_audit']):
                assert abs(a['data_energy']-b['energy']['data'])<1e-7;checks+=1
            if saved['perturbation']['failed_draws']:
                failure_records.append(dict(scene=sid,method=method,failed_draws=saved['perturbation']['failed_draws']))
        p=saved_cache[key]['predictions'][variant]
        scores=evaluate(source/'scenes'/f'{sid:04d}',p)
        assert all(r[k]==v for k,v in scores.items());checks+=1
        if variant=='inherited_2sigma':
            old=oldrows[key]
            assert all(r[k]==old[k] for k in ('qualified_tp','tp','fp','known','wrong_known'));checks+=1
        if variant=='gn_risk':
            assert all(a['decision_risk']<=.35+1e-12 for a in p['assets']);checks+=1
    active_cache={}
    for r in arows:
        sid,method,variant=r['scene'],r['method'],r['variant'];key=sid,method
        if key not in active_cache:
            path=active/'predictions'/f'{sid:04d}_{method}.json';raw=path.read_bytes();saved=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==r['prediction_sha256'];checks+=1
            active_cache[key]=saved
            assert saved['cost']['history_views']==4 and saved['cost']['additional_current_views']==2 and saved['cost']['returned'];checks+=1
            for e in saved['state']['events']:
                if e['payload_bytes']:
                    raw=(source/'scenes'/f'{sid:04d}'/'payloads'/f"{e['epoch']}_{e['view_id']:02d}.npz").read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==e['payload_sha256'];checks+=1
        p=active_cache[key]['prediction']
        prediction=p if variant=='risk' else p['terminal_alternatives'][variant]
        scores=evaluate(source/'scenes'/f'{sid:04d}',prediction)
        assert all(r[k]==v for k,v in scores.items());checks+=1
    F={(m,v):sorted([r for r in frows if r['method']==m and r['variant']==v],key=lambda x:x['scene'])
       for m in cfg['fixed_trace_methods'] for v in cfg['fixed_variants']}
    A={(m,v):sorted([r for r in arows if r['method']==m and r['variant']==v],key=lambda x:x['scene'])
       for m,selector in cfg['active_methods'] for v in ('risk','two_sigma','point')}
    summaries_fixed={m:{v:aggregate(F[m,v]) for v in cfg['fixed_variants']} for m in cfg['fixed_trace_methods']}
    summaries_active={m:{v:aggregate(A[m,v]) for v in ('risk','two_sigma','point')} for m,s in cfg['active_methods']}
    rng=np.random.default_rng(24681);cells=np.arange(48).reshape(3,16).T
    draws=np.stack([np.concatenate([cell[rng.integers(0,3,3)] for cell in cells]) for _ in range(3000)])
    pairs=[('fixed_candidate_gn_risk_minus_inherited',F['modes_regret','gn_risk'],F['modes_regret','inherited_2sigma']),
           ('fixed_candidate_gn_risk_minus_point',F['modes_regret','gn_risk'],F['modes_regret','point']),
           ('fixed_candidate_perturb_risk_minus_gn',F['modes_regret','perturb_risk'],F['modes_regret','gn_risk']),
           ('active_candidate_minus_schur',A['candidate','risk'],A['schur','risk']),
           ('active_candidate_minus_risk',A['candidate','risk'],A['risk','risk']),
           ('active_candidate_minus_same_old_evidence_gn',A['candidate','risk'],F['modes_regret','gn_risk']),
           ('active_candidate_risk_minus_2sigma',A['candidate','risk'],A['candidate','two_sigma'])]
    comparisons={}
    for label,left,right in pairs:
        changed=np.array([r['changed'] for r in left]);diff=np.array([a['qualified_tp']-b['qualified_tp'] for a,b in zip(left,right)])
        d=diff[draws].sum(1)/changed[draws].sum(1)
        comparisons[label]=dict(completion_difference=float(diff.sum()/changed.sum()),
            exploratory_paired_95_ci=np.quantile(d,[.025,.975]).tolist())
    # Prespecified replay, independent of outcome selection.
    replay=execute_policy(source/'scenes'/'0001',*cfg['budget'],lambda port:run_selection_v3(port,'candidate_regret'))
    assert replay['prediction']==active_cache[1,'candidate']['prediction'];checks+=1
    result=dict(stage='48 reused development scenes only; no calibrated confidence/independent confirmation',
        checks=checks,fixed_trace_summaries=summaries_fixed,active_summaries=summaries_active,
        comparisons=comparisons,perturbation_reference_failures=failure_records,exclusions=[],
        old_sources_inputs_preserved=True,bootstrap=3000,multiplicity='seven exploratory comparisons; no familywise confirmation')
    (active/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=checks,active=summaries_active,comparisons=comparisons,failed_perturbations=len(failure_records)),indent=2))


if __name__=='__main__':main()
