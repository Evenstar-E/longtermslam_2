import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.joint_estimation_v2 import infer_decoupled
from scripts.joint_estimation import evaluate

def main():
    base=Path('results/joint_estimation_v1');out=Path('results/joint_estimation_v2');cfg=json.loads((base/'frozen_config.json').read_text())
    old=json.loads((base/'analysis.json').read_text());new=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in (base/'metrics.jsonl').read_text().splitlines()]+new
    index={(r['scene_id'],r['method']):r for r in rows};checks=0
    def check(v):
        nonlocal checks
        assert v;checks+=1
    for folder in [base,out]:
        for p,d in json.loads((folder/'manifest.json').read_text())['files'].items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d)
    check(len(new)==480);check(len(index)==3360)
    change=[r for r in new if r['condition']['change_fraction']>0];tp=sum(r['tp'] for r in change);fp=sum(r['fp'] for r in change);fn=sum(r['fn'] for r in change)
    summary=old['summary']+[dict(method='joint_decoupled',scenes=480,pose_success=float(np.mean([r['pose_success'] for r in new])),assignment_accuracy=float(np.mean([r['assignment_accuracy'] for r in new])),change_scene_mean_f1=float(np.mean([r['f1'] for r in change])),change_micro_f1=2*tp/(2*tp+fp+fn),false_change_fraction_on_unchanged=sum(r['false_change_on_unchanged'] for r in new)/sum(r['unchanged_visible'] for r in new),translation_median_m=float(np.median([r['translation_error_m'] for r in new])),rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in new])),runtime_median_ms=1000*float(np.median([r['seconds'] for r in new]))) ]
    rng=np.random.default_rng(12381);intervals=[]
    for key in ['pose_success','assignment_accuracy','f1']:
        ids=np.array([sid for sid in range(480) if key!='f1' or index[sid,'joint_decoupled']['condition']['change_fraction']>0]);diff=np.array([float(index[s,'joint_decoupled'][key])-float(index[s,'robust_sequential'][key]) for s in ids]);groups=[np.flatnonzero(ids//30==c) for c in sorted(set(ids//30))]
        samples=[float(diff[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
        intervals.append(dict(endpoint=key,paired_scenes=len(ids),mean_difference=float(diff.mean()),percentile95=np.quantile(samples,[.025,.975]).tolist()))
    cells=[]
    for cell in range(16):
        for method in cfg['methods']+['joint_decoupled']:
            own=[index[s,method] for s in range(cell*30,(cell+1)*30)];c=own[0]['condition']
            cells.append(dict(cell=cell,method=method,**c,pose_success=float(np.mean([r['pose_success'] for r in own])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in own])),change_mean_f1=float(np.mean([r['f1'] for r in own])) if c['change_fraction']>0 else None))
    for sid in range(0,480,30):
        scene=np.load(base/'scenes'/f'{sid:04d}.npz');obs={k:scene[k] for k in scene.files if not k.startswith('truth_')};truth={k[6:]:scene[k] for k in scene.files if k.startswith('truth_')}
        metrics=evaluate(infer_decoupled(obs,cfg['algorithm'],index[sid,'joint_decoupled']['seed']+400000),truth,cfg['evaluation'])
        for k,v in metrics.items():check(v==index[sid,'joint_decoupled'][k])
    analysis=dict(summary=summary,cells=cells,paired_bootstrap_vs_robust_sequential=intervals,checks_passed=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='Development revision on reused scenes; exploratory intervals do not correct algorithm selection or establish external generalization')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    for name,items in [('summary.csv',summary),('by_condition.csv',cells)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    names=['Sequential','Robust sequential','Joint single start','Joint v1','No history (v1)','No change gate (v1)','Decoupled v2'];colors=['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    for ax,key,title in zip(axes,['pose_success','assignment_accuracy','change_scene_mean_f1'],['Pose success: <=10 cm AND <=3 deg','Mean current-to-history identity accuracy','Mean identity-aware change F1']):
        values=[s[key] for s in summary];ax.barh(range(7),values,color=colors);ax.set_yticks(range(7),names);ax.invert_yaxis();ax.set_xlim(0,1.13);ax.set_xlabel('Fraction / F1');ax.set_title(title,fontsize=10)
        for i,v in enumerate(values):ax.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('480 paired procedural scenes; F1 on 240 change scenes. v2 developed on the same scenes.\nObservation-level SE(3) simulation; no rendered perception or external benchmark.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(revision_summary=summary[-1],paired_bootstrap=intervals,checks=checks),indent=2))

if __name__=='__main__':main()
