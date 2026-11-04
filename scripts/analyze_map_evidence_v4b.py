"""Audit conditional set decisions, unknown coverage and candidate-only gain."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.map_evidence_v4b import infer


def aggregate(rows):
    sums={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','wrong_known','known','visible','changed','unchanged')}
    return dict(cases=len(rows),**sums,f1=2*sums['tp']/max(2*sums['tp']+sums['fp']+sums['fn'],1),recall=sums['tp']/max(sums['changed'],1),false_alarm=sums['fp']/max(sums['unchanged'],1),coverage=sums['known']/max(sums['visible'],1),conditional_error=sums['wrong_known']/max(sums['known'],1),base_infeasible_cases=sum(r['base_status']==2 for r in rows),solver_timeout_calls=sum(r['timeouts'] for r in rows),sampling_accepted_median=float(np.median([r['sampling_accepted'] for r in rows])))


def main():
    out=Path('results/map_evidence_v4b');cfg=json.loads((out/'frozen_config.json').read_text());man=json.loads((out/'manifest.json').read_text());rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()];checks=0
    for key in ('source_sha256','input_sha256'):
        for p,h in man[key].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;checks+=1
    assert len(rows)==512 and len({(r['scene'],r['method']) for r in rows})==512;checks+=2
    for r in rows:
        label=np.array(r['labels']);truth=np.array(r['truth_changed']);correct=np.array(r['assignment'])==r['truth_assignment'];pc=label==1;known=label>=0
        assert r['tp']==int((pc&truth&correct).sum());assert r['fp']==int((pc&(~truth|~correct)).sum());assert r['fn']==int((truth&(~pc|~correct)).sum());assert r['wrong_known']==int((known&((pc!=truth)|~correct)).sum());checks+=4
    extras=[]
    for sid in range(128):
        joint=next(r for r in rows if r['scene']==sid and r['method']=='joint_feasibility');box=next(r for r in rows if r['scene']==sid and r['method']=='coordinate_intervals')
        a=np.array(joint['labels']);b=np.array(box['labels']);assert np.array_equal(a[b>=0],b[b>=0]);checks+=1
        for i in np.flatnonzero((a==1)&(b==-1)):extras.append(dict(scene=sid,split=joint['split'],query=int(i),true_changed=joint['truth_changed'][i],assignment_correct=joint['assignment'][i]==joint['truth_assignment'][i]))
    for sid in (0,8,64,72):
        o=json.loads((out/'cases'/f'{sid:04d}_input.json').read_text());assert not {'truth','reference_boxes','all_delta','changed'}&o.keys();checks+=1;p=infer(o,cfg,cfg['development_seed_start']+sid if sid<64 else cfg['seed_start']+sid-64)
        for m in cfg['methods']:
            old=next(r for r in rows if r['scene']==sid and r['method']==m);assert old['labels']==p['labels'][m];checks+=1
    summary={s:{m:aggregate([r for r in rows if r['split']==s and r['method']==m]) for m in cfg['methods']} for s in ('development_repaired','fresh')}
    # Paired scene-level bootstrap for candidate-only incremental benefit.
    fresh={m:sorted([r for r in rows if r['split']=='fresh' and r['method']==m],key=lambda r:r['scene']) for m in ('coordinate_intervals','joint_feasibility')};rng=np.random.default_rng(221425);idx=np.concatenate([rng.integers(c*8,(c+1)*8,size=(5000,8)) for c in range(8)],axis=1)
    def metrics(m):
        a=np.array([[r[k] for k in ('tp','changed','known','visible')] for r in fresh[m]])[idx].sum(1)
        return np.column_stack([a[:,0]/np.maximum(a[:,1],1),a[:,2]/np.maximum(a[:,3],1)])
    diff=metrics('joint_feasibility')-metrics('coordinate_intervals')
    truthaudit={}
    for split,ids in [('development_repaired',range(64)),('fresh',range(64,128))]:
        audits=[man['audits'][str(i)] for i in ids];truthaudit[split]=dict(cases=len(audits),true_state_inside=sum(a['true_state_inside'] for a in audits),identity_correct=sum(a['identity_correct'] for a in audits),worst_violation=max(a['max_constraint_violation_m'] for a in audits),outside_cases=[i for i in ids if not man['audits'][str(i)]['true_state_inside']])
    conditional={}
    for group,inside in [('true_state_inside',True),('true_state_outside',False)]:
        conditional[group]={m:aggregate([r for r in rows if r['split']=='fresh' and r['method']==m and man['audits'][str(r['scene'])]['true_state_inside']==inside]) for m in cfg['methods']} if any(man['audits'][str(i)]['true_state_inside']==inside for i in range(64,128)) else {}
    analysis=dict(summary=summary,true_model_audit=truthaudit,conditional_audit=conditional,extra_joint_change_detections=extras,candidate_minus_interval=dict(recall_difference_95_ci=np.quantile(diff[:,0],[.025,.975]).tolist(),coverage_difference_95_ci=np.quantile(diff[:,1],[.025,.975]).tolist()),audit_checks=checks,total_inference_seconds=man['seconds'],analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='All cases retained; outside-true-set audit strata are diagnostic only, not filtered primary scores. No unconditional certificate.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['One solution','Nuisance samples','Coordinate bounds','Joint feasibility'];colours=['#888888','#E69F00','#009E73','#0072B2'];fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    for ax,key,title in zip(axes,['false_alarm','recall','coverage'],['False change reports on unchanged assets','True change recall (unknowns are misses)','Resolved output coverage']):
        vals=[100*summary['fresh'][m][key] for m in cfg['methods']];ax.bar(range(4),vals,color=colours);ax.set_ylim(0,110);ax.set_xticks(range(4),names,rotation=18,fontsize=8);ax.set_ylabel('%');ax.set_title(title,fontsize=9);ax.spines[['top','right']].set_visible(False)
        for x,y in enumerate(vals):ax.text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
    fig.suptitle('64 fresh RGB-D scenes with partial/noisy historical maps\nBounded set claims depend on extraction, association and calibration validity.',fontsize=11)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
