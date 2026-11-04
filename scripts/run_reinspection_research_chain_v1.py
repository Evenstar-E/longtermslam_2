"""One entry point: audit state, actual baseline acquisition, report/cost replay.

This is a baseline integration chain; candidate and ambiguity benchmark are
explicitly missing. Never treats pipeline execution as innovation evidence.
"""
import argparse
from pathlib import Path
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.report_risk_episode_v1 import run_path
from inspection_decisions.report_risk_stopping_v1 import first_stop
from inspection_decisions.costs import prefix_cost
from inspection_decisions.graph_dataset import digest,write_json,canonical_bytes
from scripts.run_evidence_value_pilot_v1 import read,save

CFG=Path('configs/reinspection_research_chain_v1.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['status','baseline-smoke','confirm'])
    args=ap.parse_args();cfg=read(CFG)
    if args.mode=='status':
        print(cfg['candidate'])
        for stage in cfg['stages']:print(stage['id'],stage['status'])
        return
    if args.mode=='confirm':
        raise SystemExit('NOT READY: ambiguity benchmark, candidate implementation and prospective confirmation contract missing. No experiment launched.')
    out=Path(cfg['output'])
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);checks=0
    for filename in cfg['legacy_audits']:
        frozen=read(filename)
        for p,sha in frozen['sources'].items():
            assert digest(p)==sha,p
            checks+=1
        model=Path(filename).parent/'models.json'
        assert digest(model)==frozen['models_sha256'];checks+=1
    model=read(cfg['action_models'])['set_rank'];pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    root=Path(cfg['baseline_scene']);save(out/'started.json',dict(config_sha256=digest(CFG),script_sha256=digest(__file__),candidate_status=cfg['candidate']))
    def snap(graph,state,record):
        write_json(out/f"stage_{record['stage']}.json.gz",dict(graph=graph,prediction=state,decision=record),True)
    # Actual actor, legal payload acquisition, backend refit, original STOP and return.
    from inspection_decisions.component_set_episode_v1 import run_episode
    episode=run_episode(root,'set_rank',{'set_rank':model},pairs,cfg['budget'],snap)
    save(out/'observed_episode.json',episode) # BEFORE evaluation truth is accessed
    metric=evaluate(root,episode['prediction']);save(out/'terminal_metrics.json',metric)
    reference=read(cfg['reference_paths'])
    stage,_=first_stop(iter(reference['states']),'original_rank',{},None)
    assert stage==len(episode['trajectory'])-1;checks+=1
    assert canonical_bytes(reference['states'][stage]['prediction'])==canonical_bytes(episode['prediction']);checks+=1
    cost=prefix_cost(episode['events'],reference['metadata'],stage,index_bytes=episode['cost']['index_bytes'])
    assert abs(cost['total_travel_m']-episode['cost']['travel_m'])<1e-9;checks+=1
    events=[e for e in episode['events'] if e.get('epoch')]
    assert len(events)==stage+2;checks+=1
    for e in events:
        token=reference['metadata'][e['epoch']][e['view_id']]['token']
        assert digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'];checks+=1
    report=dict(status='PASS_BASELINE_ONLY',checks=checks,actual_backend_fits=episode['backend_fits'],stage=stage,
        metric=metric,cost=cost,unimplemented=['new correspondence candidate','new ambiguity challenge','independent candidate confirmation'],
        interpretation='Engineering integration verified; no candidate or innovation benefit tested',
        source_files={str(p):digest(p) for p in [CFG,Path(__file__)]})
    save(out/'baseline_smoke.json',report)
    print(report['status'],'checks',checks,'actual fits',episode['backend_fits'],'stage',stage,flush=True)

if __name__=='__main__':main()
