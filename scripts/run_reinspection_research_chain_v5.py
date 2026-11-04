import argparse
from pathlib import Path
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture,prepare
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_projective_visibility_v1 import solver
from scripts.run_evidence_value_pilot_v1 import read,save


def baseline_smoke(cfg):
    out=Path(cfg['baseline_smoke_output']);assert not out.exists();out.mkdir(parents=True)
    pilot=read('configs/projective_visibility_v1.json');frozen=read(Path(pilot['output'])/'frozen_contract.json')
    for p,sha in frozen['sources'].items():assert digest(p)==sha
    sources={p:digest(p) for p in ['configs/reinspection_research_chain_v5.json',__file__]}
    save(out/'initial_freeze.json',dict(sources=sources,pilot_sources=frozen['sources'],scope='integration replay of reused scene3; no independent performance sample'))
    ep=Episode(cfg['baseline_smoke_scene'],2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
    raw=capture(h,q,ep.metadata()['sensor']);save(out/'observed.json',dict(raw=raw,events=ep.state()['events']))
    # Reuse full frozen preparation, including benchmark diagnostics, to preserve exact tested baseline.
    backend=prepare(raw,pilot);result=solver(backend,cfg['default_baseline_method'],pilot)
    save(out/'actor_report.json',result['prediction']);save(out/'backend.json',backend)
    cost=ep.finish();save(out/'cost.json',cost)
    metric=evaluate(cfg['baseline_smoke_scene'],result['prediction']);save(out/'evaluation.json',metric)
    target=Path(pilot['output']);assert canonical_bytes(result)==canonical_bytes(read(target/'0003_independent_censored.json'))
    assert canonical_bytes(backend)==canonical_bytes(read(target/'0003_backend.json'))
    assert metric==next(r['metric'] for r in read(target/'analysis.json')['rows'] if r['scene']==3 and r['method']=='independent_censored')
    assert cost['closed'] and cost['returned']
    for p,sha in sources.items():assert digest(p)==sha
    save(out/'audit.json',dict(status='PASS',scope='baseline integration replay, NOT fresh independent trial',raw_API_paths=1,backend_reconstructions=1,
        prediction_matches_frozen_pilot=True,backend_matches_frozen_pilot=True,metric_matches_frozen_pilot=True,returned=True,
        method=cfg['default_baseline_method'],qualified=metric['qualified_tp'],fp=metric['fp'],unknown=sum(a['label']<0 for a in result['prediction']['assets'])))
    print('BASELINE SMOKE PASS',dict(qualified=metric['qualified_tp'],fp=metric['fp'],task_loss=metric['task_loss'],returned=True),'reused scene3, not new performance sample')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','pilot','audit','summary','baseline-smoke','confirm']);args=ap.parse_args()
    cfg=read('configs/reinspection_research_chain_v5.json')
    if args.stage=='status':print(cfg)
    elif args.stage=='pilot':
        from scripts.run_projective_visibility_v1 import main as run
        run()
    elif args.stage=='audit':
        from scripts.audit_projective_visibility_v1 import main as audit
        audit()
    elif args.stage=='summary':
        from scripts.summarize_projective_visibility_v1 import main as summary
        summary()
    elif args.stage=='baseline-smoke':baseline_smoke(cfg)
    else:raise SystemExit('NOT READY: ordinary same-censor method matches candidate; no joint innovation gain, no confirmation launched.')


if __name__=='__main__':main()
