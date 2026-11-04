"""Task-level scoring and scene-paired mainline audit of shared-pose candidates."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.shared_pose_change_v3 import solve


def aggregate(rows):
    sums={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','tn','visible','changed_visible','unchanged_visible','false_change_on_unchanged','known_count','motion_success')}
    errs=[e for r in rows for e in r['correct_positive_motion_errors']]
    slots=sum(r['condition']['changed_objects'] for r in rows)
    return dict(cases=len(rows),**sums,changed_slots=slots,f1=2*sums['tp']/max(2*sums['tp']+sums['fp']+sums['fn'],1),recall=sums['tp']/max(sums['changed_visible'],1),false_alarm=sums['false_change_on_unchanged']/max(sums['unchanged_visible'],1),static_scene_false_positives=sum(r['fp'] for r in rows if r['condition']['changed_objects']==0),coverage=sums['known_count']/max(sums['visible'],1),identity=float(np.mean([r['assignment_accuracy'] for r in rows])),pose_success=float(np.mean([r['pose_success'] for r in rows])),median_translation_m=float(np.median([r['translation_error_m'] for r in rows])),median_rotation_deg=float(np.median([r['rotation_error_deg'] for r in rows])),motion_success_rate=sums['motion_success']/max(sums['changed_visible'],1),motion_success_all_slots=sums['motion_success']/max(slots,1),conditional_motion_median_m=float(np.median(errs)) if errs else None,conditional_motion_p90_m=float(np.quantile(errs,.9)) if errs else None,median_seconds=float(np.median([r['seconds'] for r in rows])))


def main():
    out=Path('results/shared_pose_change_v3');cfg=json.loads((out/'frozen_config.json').read_text());bcfg=json.loads((out/'base_config.json').read_text());mcfg=json.loads((out/'motion_config.json').read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()];checks=0
    for key in ('source_sha256','input_sha256'):
        for p,h in manifest[key].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;checks+=1
    assert len(rows)==324 and len({(r['scene'],r['method']) for r in rows})==324;checks+=2
    if 'recovery' in manifest:
        rec=manifest['recovery'];prefix=(out/'metrics.jsonl').read_bytes()[:rec['preserved_prefix_bytes']]
        assert hashlib.sha256(prefix).hexdigest()==rec['preserved_prefix_sha256'];checks+=1
        for case in rec['adapted_cases']:
            assert case['evaluator_mixed_asset_groups']==0
            info=manifest['base_predictions'][str(case['scene'])];obs=load_input(Path(info['input_path']))
            assert len(obs['query'])==case['adapted_detections'];checks+=2
    for r in rows:
        pc=np.array(r['prediction_changed']);tc=np.array(r['truth_changed']);correct=np.array(r['prediction_assignment'])==r['truth_assignment'];known=np.array(r['known']);errors=np.linalg.norm(np.array(r['delta'])-r['truth_delta'],axis=1)
        assert r['tp']==int((pc&tc&correct).sum());assert r['fp']==int((pc&(~tc|~correct)).sum());assert r['fn']==int((tc&(~pc|~correct)).sum());assert r['motion_success']==int((pc&tc&correct&(errors<=.1)).sum());assert r['known_count']==int(known.sum());assert r['prediction_assignment']==manifest['base_predictions'][str(r['scene'])]['base']['prediction_assignment'];checks+=6
        if r['method']!='frozen_v1':assert r['selected_start']==int(np.argmin(r['hypothesis_scores']));checks+=1
    # Replay the first scene in each split, all methods, not outcome selected.
    for split in ('diagnostic_reused','fresh'):
        sid=min(r['scene'] for r in rows if r['split']==split);info=manifest['base_predictions'][str(sid)];obs=load_input(Path(info['input_path']))
        for method in cfg['methods']:
            old=next(r for r in rows if r['scene']==sid and r['method']==method);new=solve(obs,info['base'],bcfg,mcfg,cfg,method,info['seed'])
            np.testing.assert_allclose(new['T'],old['T'],atol=1e-8);np.testing.assert_allclose(new['delta'],old['delta'],atol=1e-8);np.testing.assert_array_equal(new['changed'],old['prediction_changed']);checks+=3
    summary={split:{m:aggregate([r for r in rows if r['split']==split and r['method']==m]) for m in cfg['methods']} for split in ('diagnostic_reused','fresh')}
    fresh={m:sorted([r for r in rows if r['split']=='fresh' and r['method']==m],key=lambda r:r['scene']) for m in cfg['methods']};rng=np.random.default_rng(221024);idx=np.concatenate([rng.integers(c*8,(c+1)*8,size=(5000,8)) for c in range(8)],axis=1);contrasts={}
    def bootstrap(m):
        a=np.array([[r[k] for k in ('tp','fp','fn','motion_success','changed_visible','pose_success')] for r in fresh[m]])[idx].sum(axis=1)
        return np.column_stack([2*a[:,0]/np.maximum(2*a[:,0]+a[:,1]+a[:,2],1),a[:,3]/np.maximum(a[:,4],1),a[:,5]/64])
    joint=bootstrap('joint_profile')
    for control in cfg['methods'][:-1]:
        d=joint-bootstrap(control);contrasts[control]={name:np.quantile(d[:,j],[.025,.975]).tolist() for j,name in enumerate(('f1_difference_95_ci','motion_success_difference_95_ci','pose_success_difference_95_ci'))}
    trace=[r for r in rows if r['old_scene']==155]
    (out/'scene155_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    analysis=dict(summary=summary,fresh_paired_stratified_bootstrap=contrasts,resamples=5000,audit_checks=checks,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),engineering_recovery=manifest.get('recovery'),limits=cfg['limits'])
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['split','method']+list(summary['fresh']['frozen_v1']));w.writeheader()
        for split,d in summary.items():
            for m,v in d.items():w.writerow(dict(split=split,method=m,**v))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Frozen v1','Ray gate + surface','Ray gate + bidirectional','Shared profile'];colours=['#888888','#E69F00','#009E73','#0072B2'];fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for ax,key,title in zip(axes.ravel(),['f1','false_alarm','pose_success','motion_success_rate'],['Correct-asset change F1','Unchanged-object false alarm rate','Pose within 0.1 m / 2 deg','Correct change + motion error <= 0.1 m']):
        values=[summary['fresh'][m][key]*100 for m in cfg['methods']];ax.bar(range(4),values,color=colours);ax.set_xticks(range(4),names,rotation=12,fontsize=8);ax.set_ylim(0,110);ax.set_ylabel('%');ax.set_title(title,fontsize=10);ax.spines[['right','top']].set_visible(False)
        for x,y in enumerate(values):ax.text(x,y+1,f'{y:.1f}',ha='center',fontsize=9)
    fig.suptitle('64 fresh procedural RGB-D scenes; 4 matched-input methods\nAssignments fixed; at least 3 stable assets assumed; no reference-shape nuisance inference.',fontsize=11)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
