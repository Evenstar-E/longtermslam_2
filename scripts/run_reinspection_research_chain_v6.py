import argparse
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture
from inspection_decisions.se3_visibility_v1 import run_methods,readout
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_se3_visibility_v1 import CFG,cases,scene_root
from scripts.run_evidence_value_pilot_v1 import read,save


def smoke(chain):
    cfg=read(CFG);out=Path(chain['smoke_output']);assert not out.exists();out.mkdir(parents=True);source=Path(cfg['output']);sid=chain['smoke_case'];case=cases(cfg)[sid];root=scene_root(source,case['family'],case['variant'])
    frozen=read(source/'frozen_contract.json');assert all(digest(p)==sha for p,sha in frozen['sources'].items())
    sources={p:digest(p) for p in ['configs/reinspection_research_chain_v6.json',__file__]};save(out/'initial_freeze.json',dict(sources=sources,main_sources=frozen['sources'],scope='reused first repeated/SE3/one-history changed case25; engineering replay, not new sample'))
    ep=Episode(root,case['history_count'],2);hs=[ep.act('retrieve_history',v) for v in cfg['history_views'][:case['history_count']]];qs=[ep.preview()]+[ep.act('observe_current',v) for v in cfg['current_views'][1:]]
    raw=capture(hs,qs,ep.metadata()['sensor']);save(out/'observed.json',dict(raw=raw,events=ep.state()['events']));b=run_methods(raw,cfg);options=[]
    for k,branch in enumerate(b['registration']['retained']):
        if k==0:r=b['outputs']['se3_rescore']
        elif k==b['late_selected']:r=b['outputs']['se3_late_visibility']
        else:r=readout(raw,np.array(branch['G']),branch['matches'],cfg)
        options.append(r)
    selected=min(range(len(options)),key=lambda k:options[k]['best']['score']);r=options[selected]
    save(out/'actor_report.json',r['prediction']);save(out/'selected_result.json',r);save(out/'backend.json',b)
    cost=ep.finish();save(out/'cost.json',cost);metric=evaluate(root,r['prediction']);save(out/'evaluation.json',metric)
    target=read(Path(chain['budget_results'])/f'{sid:04d}_controls.json')['ordinary_prior_score']
    assert canonical_bytes(r)==canonical_bytes(target['result']);assert selected==target['selected'];assert cost['returned']
    assert canonical_bytes(b)==canonical_bytes(read(source/f'{sid:04d}_backend.json'))
    assert all(digest(p)==sha for p,sha in sources.items())
    save(out/'audit.json',dict(status='PASS',scope='reused case integration only',selected=selected,prediction_equals_budget_control=True,backend_equals_frozen_pilot=True,returned=True,
        qualified=metric['qualified_tp'],fp=metric['fp'],unknown=sum(a['label']<0 for a in r['prediction']['assets'])))
    print('SE3 BASELINE SMOKE PASS',dict(qualified=metric['qualified_tp'],fp=metric['fp'],loss=metric['task_loss'],returned=True),'reused case, not new independent trial')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','pilot','audit','summary','baseline-smoke','confirm']);args=ap.parse_args();cfg=read('configs/reinspection_research_chain_v6.json')
    if args.stage=='status':print(cfg)
    elif args.stage=='pilot':
        from scripts.run_se3_visibility_v1 import main as run
        run()
    elif args.stage=='audit':
        from scripts.audit_se3_visibility_v1 import main as audit
        audit()
    elif args.stage=='summary':
        from scripts.summarize_se3_visibility_v1 import main as summarize
        summarize()
    elif args.stage=='baseline-smoke':smoke(cfg)
    else:raise SystemExit('NOT READY: matched-compute ordinary method explains candidate gain; real identity alternatives only verified in separate input fixture, not active task evaluation.')


if __name__=='__main__':main()
