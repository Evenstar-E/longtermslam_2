"""Extra correctness check: first candidate family with a strict early STOP.

Scene0 prospective replay stopped at budget. This adds a deterministic first-ID
early-prefix physical check without changing any policy/model/primary outcome.
"""
from pathlib import Path
from unittest.mock import patch
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.report_risk_episode_v1 import run_path
from inspection_decisions.optional_stopping_v1 import decision
from inspection_decisions.graph_dataset import canonical_bytes,digest
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths


def main():
    out=Path('results/optional_stopping_v1');cfg=read('configs/optional_stopping_v1.json')
    if (out/'early_stop_replay_audit.json').exists():raise FileExistsError('already validated')
    a=read(out/'analysis.json');b=read(out/'models.json');freeze=read(out/'frozen_models.json')
    for p,sha in freeze['sources'].items():assert digest(p)==sha,p
    check_count=0
    def check(ok,message):
        nonlocal check_count
        assert ok,message
        check_count+=1
    row=min((r for r in a['rows'] if r['method']=='optional_cf_temporal' and r['stage']<5),key=lambda r:r['scene'])
    sid=row['scene'];root=out/f'evaluation_{sid%2}'/'scenes'/f'{sid:04d}'
    path=next(p for p in load_paths(out,'evaluation') if p['scene']==sid and p['policy']=='rank')
    def stop(method,state,previous,initial,models,setting,price):return decision(method,state,previous,initial,models,setting)
    with patch('inspection_decisions.report_risk_episode_v1.should_stop',stop):
        actual=run_path(root,'rank',read(cfg['action_model'])['set_rank'],read('configs/evidence_graph_dataset_v1.json')['pairs'],cfg['budget'],lambda g,s:None,
            stopper=dict(method='optional_cf_temporal',models=b,setting=b['settings']['optional_cf_temporal'],price=cfg['price']))
    save(out/'physical_early_stop_replay.json',actual)
    check(actual['backend_fits']==row['stage']+1,'early stop stage')
    for state in actual['states']:
        t=state['stage'];check(canonical_bytes(state['prediction'])==canonical_bytes(path['observed']['states'][t]['prediction']),'physical prefix report')
        chosen=next(d for d in row['decisions'] if d['stage']==t)
        check(canonical_bytes(state['stop_decision'])==canonical_bytes({k:v for k,v in chosen.items() if k not in ('stage','stop')}),'physical prefix decision')
    events=[e for e in actual['events'] if e.get('epoch')];reference=[e for e in path['observed']['events'] if e.get('epoch')]
    check(len(events)==row['stage']+2,'no future payloads read')
    for e,r in zip(events,reference):check(e['payload_sha256']==r['payload_sha256'] and e['epoch']==r['epoch'] and e['view_id']==r['view_id'],'prefix payload')
    check(abs(actual['cost']['travel_m']-row['cost']['total_travel_m'])<1e-10,'physical early return')
    check(actual['cost']['payload_bytes']==row['cost']['payload_bytes'],'prefix bytes')
    check(canonical_bytes(evaluate(root,actual['states'][-1]['prediction']))==canonical_bytes(row['metric']),'physical endpoint metric')
    save(out/'early_stop_replay_audit.json',dict(status='PASS',scene=sid,stage=row['stage'],checks=check_count,
        selection='POSTHOC validation only: smallest scene ID with frozen candidate early STOP; no policy tuning',
        additional_backend_fits=actual['backend_fits'],script_sha256=digest(__file__),
        total_new_backend_fits=a['actual_evaluation_fits']+a['actual_validation_fits']+actual['backend_fits']))
    print('Physical early STOP PASS',sid,row['stage'],'checks',check_count,'extra fits',actual['backend_fits'],flush=True)


if __name__=='__main__':main()
