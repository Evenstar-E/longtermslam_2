"""POSTHOC resource-matched controls, not a prospective confirmation experiment."""
from pathlib import Path
import copy
import numpy as np
from comparative_evidence.estimation_v1 import world_components
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import decode
from inspection_decisions.se3_visibility_v1 import readout
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_se3_visibility_v1 import CFG,cases,scene_root


def main():
    cfg=read(CFG);source=Path(cfg['output']);out=Path('results/se3_visibility_budget_audit_v1');assert not out.exists();out.mkdir(parents=True)
    sources={__file__:digest(__file__),CFG:digest(CFG),'inspection_decisions/se3_visibility_v1.py':digest('inspection_decisions/se3_visibility_v1.py')}
    save(out/'diagnostic_contract.json',dict(status='POSTHOC budget-confound diagnostic after original30 observed; not prospective claim',sources=sources,
        controls=dict(ordinary_prior_score='min final common sparse-motion best.score over same retained poses',ordinary_pixel_score='min raw projected fit losses +12 per unmatched current observation over same retained poses'),
        no_tuned_parameters=True,no_new_worlds=True,original_results_unchanged=True))
    rows=[];extra_readouts=0;checks=0;branch_rows=[]
    for sid,case in enumerate(cases(cfg)):
        b=read(source/f'{sid:04d}_backend.json');raw=read(source/f'{sid:04d}_observed.json')['raw'];options=[]
        for k,branch in enumerate(b['registration']['retained']):
            if k==0:r=b['outputs']['se3_rescore']
            elif k==b['late_selected']:r=b['outputs']['se3_late_visibility']
            else:r=readout(raw,np.array(branch['G']),branch['matches'],cfg);extra_readouts+=1
            options.append(r)
        checks+=1;assert all(abs(r['late_score']-b['late_scores'][k])<1e-9 for k,r in enumerate(options))
        total=sum(len(world_components(decode(o))) for o in raw['current'])
        scores=dict(ordinary_prior_score=[r['best']['score'] for r in options],
            ordinary_pixel_score=[sum(v['fit']['loss'] for v in r['projective_fits'])+12*(total-sum(len(p) for p in r['best']['matches'])) for r in options])
        results={}
        for method,values in scores.items():
            selected=min(range(len(options)),key=lambda k:values[k]);results[method]=dict(selected=selected,scores=values,result=copy.deepcopy(options[selected]))
        save(out/f'{sid:04d}_options.json',dict(options=options,scores=scores))
        save(out/f'{sid:04d}_controls.json',results)
        # New control predictions saved before evaluator access; definitions remain POSTHOC.
        for method,v in results.items():
            r=v['result'];rows.append(dict(scene=sid,**case,method=method,selected=v['selected'],metric=evaluate(scene_root(source,case['family'],case['variant']),r['prediction']),
                unknown=sum(a['label']<0 for a in r['prediction']['assets'])))
        branch_rows.append(dict(scene=sid,retained=len(options),retained_actual_assignments=len({str(v['matches']) for v in b['registration']['retained']}),
            candidate_selected=b['late_selected'],control_selected={m:v['selected'] for m,v in results.items()}))
    original=read(source/'analysis.json');summaries={}
    for condition in ['repeat_se3_h1','all']:
        summaries[condition]={}
        for method in ['ordinary_prior_score','ordinary_pixel_score']:
            g=[r for r in rows if r['method']==method and (condition=='all' or r['condition']==condition)]
            v={k:sum(r['metric'][k] for r in g) for k in ['qualified_tp','fp','bad_motion_reports','identity_wrong_references','task_loss']};v['unknown']=sum(r['unknown'] for r in g);summaries[condition][method]=v
    rng=np.random.default_rng(2324026);ix=rng.integers(0,6,(5000,6));contrasts={}
    for method in ['ordinary_prior_score','ordinary_pixel_score']:
        d=np.array([next(r['metric']['task_loss'] for r in original['rows'] if r['condition']=='repeat_se3_h1' and r['family']==family and r['method']=='se3_late_visibility')-
            next(r['metric']['task_loss'] for r in rows if r['condition']=='repeat_se3_h1' and r['family']==family and r['method']==method) for family in range(6,12)])
        contrasts[method]=dict(mean_candidate_minus_control=float(d.mean()),exploratory_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),differences=d.tolist())
    assert all(digest(p)==sha for p,sha in sources.items())
    save(out/'analysis.json',dict(status='POSTHOC_BUDGET_AUDIT',rows=rows,summaries=summaries,contrasts=contrasts,branch_rows=branch_rows,
        extra_projective_readouts=extra_readouts,new_API_paths=0,new_worlds=0,control_outputs=60,checks=checks,exclusions=[]))
    print('POSTHOC BUDGET AUDIT',summaries,'contrasts',contrasts,'extra readouts',extra_readouts,flush=True)


if __name__=='__main__':main()
