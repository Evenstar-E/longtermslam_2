"""Audit frozen results and post-hoc detector ablations on identical saved poses."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.joint_estimation import transform,evaluate
from scripts.partial_surface_estimation import signed_box_distance,infer


def load_input(path):
    a=dict(np.load(path));points=np.split(a['points'],np.cumsum(a['lengths'])[:-1]);pixels=np.split(a['query_pixels'],np.cumsum(a['pixel_lengths'])[:-1])
    a['query']=[dict(points=p,feature=f,all_pixels=pix) for p,f,pix in zip(points,a['query_features'],pixels)]
    return a


def aggregate(rows):
    s={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','tn','unchanged_visible','false_change_on_unchanged','changed_visible','visible','known_count')}
    s.update(cases=len(rows),f1=2*s['tp']/max(2*s['tp']+s['fp']+s['fn'],1),recall=s['tp']/max(s['changed_visible'],1),false_alarm=s['false_change_on_unchanged']/max(s['unchanged_visible'],1),coverage=s['known_count']/max(s['visible'],1),pose_success=np.mean([r['pose_success'] for r in rows]),identity=np.mean([r['assignment_accuracy'] for r in rows]),median_translation_m=np.median([r['translation_error_m'] for r in rows]),median_rotation_deg=np.median([r['rotation_error_deg'] for r in rows]),median_ms=np.median([r['seconds']*1000 for r in rows]))
    return {k:float(v) if isinstance(v,np.floating) else v for k,v in s.items()}


def main():
    out=Path('results/partial_surface_v1');cfg=json.loads((out/'frozen_config.json').read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
    checks=0;delta_errors=[];boundary_fits=0
    for p,h in manifest['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;checks+=1
    assert len(rows)==128*5 and len({(r['scene'],r['method']) for r in rows})==640;checks+=2
    ablations=[]
    for sid in range(128):
        obs=load_input(out/'cases'/f'{sid:04d}_input.npz');truth=dict(np.load(out/'cases'/f'{sid:04d}_truth.npz'))
        assert not {'T','assignment','changed','boxes','current_boxes','delta'}&obs.keys();checks+=1
        assert len(obs['query'])==len(truth['assignment']);checks+=1
        for r in [r for r in rows if r['scene']==sid]:
            A=np.array(r['prediction_assignment']);pc=np.array(r['prediction_changed']);known=np.array(r['known']);tc=truth['changed'];correct=A==truth['assignment']
            assert r['tp']==int((pc&tc&correct).sum());assert r['fp']==int((pc&(~tc|~correct)).sum());assert r['fn']==int((tc&(~pc|~correct)).sum());assert r['known_count']==int(known.sum());checks+=4
            if r['method']=='surface_joint':
                estimated=np.array(r['estimated_delta']);selected=pc&tc&correct
                delta_errors.extend(np.linalg.norm(estimated[selected]-truth['delta'][selected],axis=1).tolist())
                boundary_fits+=int((np.max(np.abs(estimated[selected]),axis=1)>.699).sum())
            if r['method'] in ('surface_robust','surface_joint'):
                score=np.array([np.median(np.abs(signed_box_distance(transform(q['points'],np.array(r['T'])),obs['bounds'][j]))) for q,j in zip(obs['query'],A)])
                pred=dict(T=np.array(r['T']),assignment=A,changed=(score>cfg['change_threshold_m'])&known)
                ev=evaluate(pred,truth,dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=2))
                ablations.append(dict(scene=sid,cell=r['cell'],condition=r['condition'],method=r['method']+'_sdf_only',known_count=r['known_count'],seconds=r['seconds'],**ev))
        # Reproduce four preselected cases directly from serialized inference input.
        if sid in (0,32,64,96):
            old=next(r for r in rows if r['scene']==sid and r['method']=='surface_joint');pred=infer(obs,cfg,'surface_joint')
            np.testing.assert_allclose(pred['T'],old['T'],atol=1e-10);np.testing.assert_allclose(pred['residual_m'],old['residual_m'],atol=1e-10);checks+=2
    (out/'score_ablations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ablations),encoding='utf-8')
    methods=cfg['methods']+[m+'_sdf_only' for m in ('surface_robust','surface_joint')];allrows=rows+ablations
    summary={m:aggregate([r for r in allrows if r['method']==m]) for m in methods}
    cells={str(c):{m:aggregate([r for r in allrows if r['method']==m and r['cell']==c]) for m in methods} for c in range(8)}
    # Stratified scene bootstrap, preserving factorial cell sizes and paired methods.
    base=np.array([[r[k] for k in ('tp','fp','fn')] for r in rows if r['method']=='surface_robust'])
    joint=np.array([[r[k] for k in ('tp','fp','fn')] for r in rows if r['method']=='surface_joint'])
    rng=np.random.default_rng(221022);indices=np.concatenate([rng.integers(c*16,(c+1)*16,size=(10000,16)) for c in range(8)],axis=1)
    def f1(x):
        a=x[indices].sum(axis=1);return 2*a[:,0]/np.maximum(2*a[:,0]+a[:,1]+a[:,2],1)
    difference=f1(joint)-f1(base);ci=np.quantile(difference,[.025,.975]).tolist()
    analysis=dict(summary=summary,cells=cells,paired_pooled_f1_difference=dict(contrast='surface_joint minus surface_robust',estimate=summary['surface_joint']['f1']-summary['surface_robust']['f1'],stratified_scene_bootstrap_95_ci=ci,resamples=10000),local_translation_diagnostic=dict(correct_positive_count=len(delta_errors),median_vector_error_m=float(np.median(delta_errors)),p90_vector_error_m=float(np.quantile(delta_errors,.9)),near_bound_count=boundary_fits,note='Post-pose one-sided local fitting, not a fully joint deformation solver; partial-surface translation may be ambiguous.'),audit_checks=checks,score_ablation_status='Posthoc diagnostic: identical saved poses, assignments and full-model known gates; no new independent scenes; same threshold. Only change score loses reference ray evidence.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['method']+list(summary[methods[0]]));writer.writeheader();writer.writerows(dict(method=m,**summary[m]) for m in methods)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Centroid v4','Surface robust','Surface group','Initial pose','Oracle pose'];x=np.arange(5)
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained');colours=['#888888','#E69F00','#0072B2','#AA4499','#009E73']
    for ax,key,title in zip(axes.ravel(),['false_alarm','recall','pose_success','coverage'],['False alarms on unchanged objects (lower better)','Correct-asset change recall (higher better)','Pose within 0.1 m / 2 deg (higher better)','Testable output coverage (unknowns retained)']):
        values=[summary[m][key]*100 for m in cfg['methods']];ax.bar(x,values,color=colours);ax.set_xticks(x,names,rotation=15);ax.set_ylabel('%');ax.set_ylim(0,110);ax.set_title(title,fontsize=10);ax.spines[['right','top']].set_visible(False)
        for a,b in zip(x,values):ax.text(a,b+1,f'{b:.1f}',ha='center',fontsize=9)
    fig.suptitle('128 procedural RGB-D scenes; 8 conditions × 16 seeds\nOracle receives true pose. Group method is a development candidate, not established novelty.',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
