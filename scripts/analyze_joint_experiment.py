"""Paired scene-level analysis, all cells retained, no parameter selection."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.joint_estimation import infer,evaluate

def main():
    out=Path('results/joint_estimation_v1');cfg=json.loads((out/'frozen_config.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];checks=0
    def check(x):
        nonlocal checks
        assert x;checks+=1
    for p,d in manifest['files'].items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d)
    check(len(rows)==480*6);index={(r['scene_id'],r['method']):r for r in rows};check(len(index)==len(rows))
    summaries=[]
    def aggregate(own):
        changed=[r for r in own if r['condition']['change_fraction']>0];tp=sum(r['tp'] for r in changed);fp=sum(r['fp'] for r in changed);fn=sum(r['fn'] for r in changed)
        return dict(scenes=len(own),pose_success=float(np.mean([r['pose_success'] for r in own])),assignment_accuracy=float(np.mean([r['assignment_accuracy'] for r in own])),
           change_scene_mean_f1=float(np.mean([r['f1'] for r in changed])) if changed else None,
           change_micro_f1=2*tp/max(2*tp+fp+fn,1) if changed else None,
           false_change_fraction_on_unchanged=sum(r['false_change_on_unchanged'] for r in own)/sum(r['unchanged_visible'] for r in own),
           translation_median_m=float(np.median([r['translation_error_m'] for r in own])),rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in own])),
           runtime_median_ms=1000*float(np.median([r['seconds'] for r in own])))
    for method in cfg['methods']:summaries.append(dict(method=method,**aggregate([r for r in rows if r['method']==method])))
    cells=[]
    for cell in range(16):
        for method in cfg['methods']:
            own=[r for r in rows if r['scene_id']//cfg['seeds']==cell and r['method']==method]
            check(len(own)==30);cells.append(dict(cell=cell,condition=own[0]['condition'],method=method,**aggregate(own)))
    rng=np.random.default_rng(cfg['evaluation']['paired_scene_bootstrap_seed']);intervals=[]
    for endpoint in ['pose_success','f1','assignment_accuracy']:
        eligible=[sid for sid in range(480) if endpoint!='f1' or index[sid,'joint_history']['condition']['change_fraction']>0]
        ids=np.array(eligible);diff=np.array([float(index[s,'joint_history'][endpoint])-float(index[s,'robust_sequential'][endpoint]) for s in ids])
        groups=[np.flatnonzero(ids//30==c) for c in sorted(set(ids//30))]
        samples=[]
        for _ in range(cfg['evaluation']['bootstrap_samples']):
            selected=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);samples.append(float(diff[selected].mean()))
        intervals.append(dict(comparison='joint_history minus robust_sequential',endpoint=endpoint,paired_scenes=len(ids),mean_difference=float(diff.mean()),percentile95=np.quantile(samples,[.025,.975]).tolist(),unit='scene',stratified_by_condition=True))
    # Re-evaluate saved scenes in every factorial cell. No raw truth is passed into infer.
    for cell in range(16):
        sid=cell*30;scene=np.load(out/'scenes'/f'{sid:04d}.npz');obs={k:scene[k] for k in scene.files if not k.startswith('truth_')};truth={k[6:]:scene[k] for k in scene.files if k.startswith('truth_')}
        for method in cfg['methods']:
            pred=infer(obs,cfg['algorithm'],method,index[sid,method]['seed']+400000);got=evaluate(pred,truth,cfg['evaluation'])
            for key,value in got.items():check(value==index[sid,method][key])
    analysis=dict(summary=summaries,cells=cells,paired_bootstrap=intervals,checks_passed=checks,
       uncertainty='Exploratory paired scene bootstrap stratified by factorial condition. Procedural scene seeds,not independent real environments. No multiplicity-adjusted confirmatory test.',
       analyzed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with (out/'by_condition.csv').open('w',newline='',encoding='utf-8') as f:
        flat=[{**{k:v for k,v in r.items() if k!='condition'},**r['condition']} for r in cells];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained');short=['Sequential','Robust sequential','Joint single start','Joint + history','Joint no history','Joint no change gate'];colors=['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
    for ax,key,title in zip(axes,['pose_success','assignment_accuracy','change_scene_mean_f1'],['SE(3) pose success (480 scenes)','Mean identity accuracy (480 scenes)','Mean identity-aware change F1 (240 scenes)']):
        ax.barh(np.arange(6),[s[key] for s in summaries],color=colors);ax.set_yticks(np.arange(6),short);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title);ax.set_xlabel('Fraction / F1')
        for i,s in enumerate(summaries):ax.text(min(s[key]+.01,.94),i,f'{s[key]:.3f}',va='center',fontsize=9)
    fig.suptitle('Observation-level simulation: 16 conditions x 30 seeds; no rendered perception');fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    # Raw scene differences displayed without treating object measurements as replicates.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,title in zip(axes,['pose_success','f1'],['Pose success difference','Change F1 difference (change scenes only)']):
        vals=[]
        for cell in range(16):
            if key=='f1' and index[cell*30,'joint_history']['condition']['change_fraction']==0:continue
            d=[float(index[s,'joint_history'][key])-float(index[s,'robust_sequential'][key]) for s in range(cell*30,(cell+1)*30)]
            ax.scatter(np.full(30,cell)+np.linspace(-.18,.18,30),d,s=10,alpha=.45,color='#0072B2');ax.plot(cell,np.mean(d),'D',color='black',markersize=5);vals.extend(d)
        ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Factorial condition ID',ylabel='Joint minus robust sequential',title=title,ylim=(-1.08,1.08))
    fig.savefig(out/'paired_differences.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=summaries,paired_bootstrap=intervals,checks=checks),indent=2))

if __name__=='__main__':main()
