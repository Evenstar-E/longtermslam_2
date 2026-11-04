import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.history_alignment_control import infer
from scripts.joint_estimation import evaluate

def main():
    out=Path('results/history_alignment_control_v1');base=Path('results/history_modes_v1');plan=json.loads((base/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];index={(r['base_scene'],r['stress']):r for r in rows};checks=0
    def check(v):
        nonlocal checks
        assert v;checks+=1
    for path,d in manifest.items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==d)
    check(len(rows)==512);check(len(index)==512);summary=[]
    for level,s in enumerate(plan['stresses']):
        rs=[r for r in rows if r['stress']==s['name']];c=[r for r in rs if ((r['base_scene']//16)//4)%2]
        summary.append(dict(stress=s['name'],method='history_alignment',cases=len(rs),pose_success=float(np.mean([r['pose_success'] for r in rs])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in rs])),change_f1=float(np.mean([r['f1'] for r in c])),qualified_change_f1=float(np.mean([r['qualified_f1'] for r in c])),reference_valid_fraction=float(np.mean([r['reference_valid_fraction'] for r in rs])),false_change_rate=sum(r['false_change_on_unchanged'] for r in rs)/sum(r['unchanged_visible'] for r in rs),median_runtime_ms=1000*float(np.median([r['seconds'] for r in rs]))))
        for sid in range(0,256,16):
            a=np.load(base/'cases'/f'{level}_{sid:04d}.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k:a['truth_'+k] for k in ['T','assignment','changed']};pred=infer(obs,cfg['algorithm'],sampler,plan['seed_base']+sid+400000)
            for k,v in evaluate(pred,truth,cfg['evaluation']).items():check(v==index[sid,s['name']][k])
    merged=json.loads((base/'analysis.json').read_text())['summary']+summary
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
    (out/'analysis.json').write_text(json.dumps(dict(summary=summary,all_methods=merged,checks=checks,scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Post-hoc conventional history-realignment control on reused development cases;not a newly validated original algorithm'),indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained');styles=[('mean','Mean memory','#777777','o'),('anchor_only','Reference frame only','#56B4E9','^'),('anchor_cluster','Reference cluster','#CC79A7','v'),('v6_modes','v6 modes','#009E73','X'),('history_alignment','Historical realignment control','#D55E00','s')]
    for ax,key,title in zip(axes,['pose_success','qualified_change_f1'],['Pose success <=10 cm AND <=3 deg','Reference-qualified change F1']):
        for m,label,color,marker in styles:
            values=[next(r[key] for r in merged if r['stress']==s['name'] and r['method']==m) for s in plan['stresses']];ax.plot(range(4),values,marker=marker,color=color,label=label,lw=1.6)
        ax.set_xticks(range(4),['Clean','Minority mixing','Majority mixing','Mixing + pose'],rotation=15);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('Fraction / F1');ax.grid(alpha=.2)
    axes[1].legend(loc='lower left',fontsize=8);fig.suptitle('All original methods retained; realignment added after the v6 diagnostic\n128 RGB-D base scenes, validated complete first reference; historical repair uses no current data.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary,checks=checks),indent=2))

if __name__=='__main__':main()
