"""Exploratory factorial analysis, geometry audit and saved prediction replay."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from comparative_evidence.evaluation_v1 import score_prediction
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v2 import run_selection_v2


def main():
    out=Path('results/comparative_selection_v2');cfg=json.loads((out/'frozen_config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in Path('results/comparative_selection_v1/metrics.jsonl').read_text().splitlines()]
    checks=0
    for p,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p;checks+=1
    assert len(rows)==432 and len({(r['scene'],r['method']) for r in rows})==432;checks+=1
    source=Path(cfg['source_data']);old_manifest=json.loads((source/'manifest.json').read_text())
    assert hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()==manifest['input_manifest_sha256'];checks+=1
    for scene in old_manifest['scenes']:
        case=source/'scenes'/f"{scene['scene']:04d}"
        for name,digest in scene['hashes'].items():
            assert hashlib.sha256((case/name).read_bytes()).hexdigest()==digest;checks+=1
    summaries={};geometric={};lookup={}
    for name,method,backend in cfg['experiments']:
        rs=sorted([r for r in rows if r['method']==name],key=lambda r:r['scene']);lookup[name]=rs
        s={k:sum(r[k] for r in rs) for k in ('tp','qualified_tp','fp','changed','unchanged','known','wrong_known','identity_wrong_references','assets')}
        s.update(completion=s['qualified_tp']/s['changed'],coverage=s['known']/s['assets'],
            mean_travel_m=float(np.mean([r['cost']['travel_m'] for r in rs])),
            median_policy_compute_s=float(np.median([r['timings']['policy_wall_excluding_api_s'] for r in rs])))
        chosen=Counter();effective=[];gaps=[];multi=0
        final_motion_good=0;oracle_global_mode_motion_good=0;energy_selection_misses=0
        for row in rs:
            sid=row['scene'];path=out/'predictions'/f'{sid:04d}_{name}.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prediction_sha256'];checks+=1
            saved=json.loads(path.read_text());p=saved['prediction'];root=source/'scenes'/f'{sid:04d}'
            rescore=score_prediction(root,p)
            assert all(row[k]==v for k,v in rescore.items());checks+=1
            assert saved['cost']['returned'] and saved['cost']['closed'];checks+=1
            if p['mode_audit']:
                truth=json.loads((root/'evaluation'/'truth.json').read_text())
                good_counts=[]
                for mode in p['mode_audit']:
                    good=0
                    for ownership in row['ownership_records']:
                        j=ownership['owner'];k=ownership['ref_id']
                        if j>=0 and truth['changed'][j] and ownership['identity_ok']:
                            good+=int(np.linalg.norm(np.array(mode['mean_delta'][k])-truth['delta'][j])<=.10)
                    good_counts.append(good)
                selected_index=next(i for i,m in enumerate(p['mode_audit']) if m['selected'])
                final_motion_good+=good_counts[selected_index]
                oracle_global_mode_motion_good+=max(good_counts)
                energy_selection_misses+=int(good_counts[selected_index]<max(good_counts))
            expected=cfg['full_evidence_budget'] if method=='full_evidence' else cfg['budget']
            assert [saved['cost']['history_views'],saved['cost']['additional_current_views']]==expected;checks+=1
            for event in saved['state']['events']:
                if event['payload_bytes']:
                    raw=(root/'payloads'/f"{event['epoch']}_{event['view_id']:02d}.npz").read_bytes()
                    assert len(raw)==event['payload_bytes'] and hashlib.sha256(raw).hexdigest()==event['payload_sha256'];checks+=1
            for audit in p['solver_audit']:
                modes=audit['modes'];weights=np.array([m['weight'] for m in modes]);selected=next(m for m in modes if m['selected'])
                assert abs(weights.sum()-1)<1e-10 and (weights>=0).all();checks+=1
                totals=[m['energy']['total'] for m in modes]
                if backend=='modes':
                    assert abs(selected['energy']['total']-min(totals))<1e-9;checks+=1
                chosen[selected['name']]+=1;effective.append(1/(weights@weights));multi+=int((weights>=.05).sum()>1)
                gaps.append(modes[0]['energy']['total']-selected['energy']['total'])
                for m in modes:
                    e=m['energy'];assert abs(e['total']-e['data']-e['shape']-e['motion']-e['nuisance'])<1e-8;checks+=1
        summaries[name]=s
        geometric[name]=dict(selected=dict(chosen),solver_states=len(effective),
            multi_with_two_weights_ge_005=multi,
            mean_effective_modes=float(np.mean(effective)) if effective else None,
            mean_audit_improvement_over_zero=float(np.mean(gaps)) if gaps else None,
            final_selected_motion_good=final_motion_good,
            oracle_global_mode_motion_good=oracle_global_mode_motion_good,
            final_energy_selection_misses_scenes=energy_selection_misses,
            oracle_scope='Posthoc motion-only best single mode per scene; not deployable, no label/confidence change')
    lookup['v1_old_separation']=sorted([r for r in old if r['method']=='candidate_separation'],key=lambda r:r['scene'])
    lookup['v1_entropy']=sorted([r for r in old if r['method']=='entropy_mc'],key=lambda r:r['scene'])
    lookup['v1_task_risk']=sorted([r for r in old if r['method']=='task_risk_mc'],key=lambda r:r['scene'])
    pairs=[('gaussian_regret','v1_old_separation'),('gaussian_regret','gaussian_risk_compatible'),
           ('modes_regret','modes_old_separation'),('modes_regret','modes_risk_compatible'),
           ('modes_regret','modes_entropy'),('modes_regret','modes_schur'),
           ('modes_regret','gaussian_regret'),('modes_full','icp_only_full')]
    rng=np.random.default_rng(24681);cells=np.arange(48).reshape(3,16).T
    draws=np.stack([np.concatenate([cell[rng.integers(0,3,3)] for cell in cells]) for _ in range(3000)])
    comparisons={}
    for a,b in pairs:
        ar,br=lookup[a],lookup[b];changed=np.array([r['changed'] for r in ar]);diff=np.array([x['qualified_tp']-y['qualified_tp'] for x,y in zip(ar,br)])
        boot=diff[draws].sum(axis=1)/changed[draws].sum(axis=1)
        comparisons[f'{a}_minus_{b}']=dict(gain=float(diff.sum()/changed.sum()),
            exploratory_paired_95_ci=np.quantile(boot,[.025,.975]).tolist(),
            better_scenes=int((diff>0).sum()),worse_scenes=int((diff<0).sum()))
    for sid,method,backend,name in ((1,'candidate_regret','modes','modes_regret'),(15,'task_risk_compatible_mc','gaussian','gaussian_risk_compatible')):
        replay=execute_policy(source/'scenes'/f'{sid:04d}',*cfg['budget'],lambda port:run_selection_v2(port,method,backend))
        saved=json.loads((out/'predictions'/f'{sid:04d}_{name}.json').read_text())
        assert replay['prediction']==saved['prediction'];checks+=1
    result=dict(stage='All48 reused development scenes; exploratory optimization, no independent confirmation',
        checks=checks,summaries=summaries,mode_audit=geometric,comparisons=comparisons,
        bootstrap=3000,multiplicity='eight exploratory comparisons, not familywise-confirmatory claims',
        exclusions=[],old_sources_and_inputs_preserved=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=checks,summaries=summaries,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
