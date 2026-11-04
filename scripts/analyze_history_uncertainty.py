import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_history_uncertainty import predict
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v4 import coverage

def main():
    out=Path('results/history_uncertainty_v1');plan=json.loads((out/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];index={(r['base_scene'],r['stress'],r['method']):r for r in rows};checks=0
    def check(v):
        nonlocal checks
        assert v;checks+=1
    for p,d in manifest['files'].items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d)
    check(len(rows)==1536);check(len(index)==1536)
    def changed(sid):return ((sid//16)//4)%2==1
    summary=[]
    for level,s in enumerate(plan['stress']):
        for m in plan['methods']:
            rs=[r for r in rows if r['stress']==s['name'] and r['method']==m];c=[r for r in rs if changed(r['base_scene'])]
            summary.append(dict(stress=s['name'],method=m,cases=len(rs),absolute_pose_success=float(np.mean([r['pose_success'] for r in rs])),relative_pose_success=float(np.mean([r['relative_pose_success'] for r in rs])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in rs])),change_f1=float(np.mean([r['f1'] for r in c])),median_translation_m=float(np.median([r['translation_error_m'] for r in rs])),median_relative_translation_m=float(np.median([r['relative_translation_m'] for r in rs])),median_runtime_ms=1000*float(np.median([r['seconds'] for r in rs]))))
        for sid in range(0,256,16):
            a=np.load(out/'cases'/f'{level}_{sid:04d}.npz');obs={k:a[k] for k in a.files if not k.startswith('truth_')};truth={k:a['truth_'+k] for k in ['T','assignment','changed']};relative={**truth,'T':a['truth_common']@truth['T']}
            check(np.linalg.eigvalsh(obs['history_covariances']).min()>0)
            # The shared gauge transform cannot change physical pair distances.
            pts=obs['history_positions'][0];R=a['truth_common'][:3,:3];check(np.allclose(np.linalg.norm(pts[1:]-pts[:1],axis=1),np.linalg.norm((pts[1:]-pts[:1])@R,axis=1)))
            for m in plan['methods']:
                pred=predict(obs,cfg['algorithm'],sampler,m,plan['seed_base']+sid+400000);e=evaluate(pred,truth,cfg['evaluation']);rel=evaluate(pred,relative,cfg['evaluation']);e.update(coverage(pred,relative,cfg['evaluation']),relative_pose_success=rel['pose_success'],relative_translation_m=rel['translation_error_m'])
                for key,value in e.items():check(value==index[sid,s['name'],m][key])
    ids=sorted({r['base_scene'] for r in rows});rng=np.random.default_rng(43981);intervals=[]
    for s in plan['stress']:
        for target,control in [('weighted_memory','v4'),('v5_uncertainty','weighted_memory')]:
            for endpoint in ['pose_success','f1']:
                eligible=np.array([i for i in ids if endpoint!='f1' or changed(i)]);groups=[np.flatnonzero(eligible//16==c) for c in sorted(set(eligible//16))];d=np.array([float(index[i,s['name'],target][endpoint])-float(index[i,s['name'],control][endpoint]) for i in eligible])
                draws=[float(d[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
                intervals.append(dict(stress=s['name'],comparison=target+' minus '+control,endpoint=endpoint,base_scenes=len(eligible),mean_difference=float(d.mean()),percentile95=np.quantile(draws,[.025,.975]).tolist()))
    result=dict(summary=summary,paired_bootstrap_by_stress=intervals,checks=checks,base_scenes=128,stress_cases=512,scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='Per-stress paired base-scene bootstrap stratified by original visual condition;repeated stress cases are not independent new scenes. Known covariance calibration;no full cross-object correlation optimization.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained');styles=[('v4','v4 mean history','#0072B2','o'),('weighted_memory','GLS history + v4','#E69F00','s'),('v5_uncertainty','v5 full uncertainty','#009E73','^')]
    for ax,key,title in zip(axes,['absolute_pose_success','relative_pose_success','change_f1'],['Absolute world-pose success','History-gauge-relative pose success','Identity-aware change F1']):
        for method,label,color,marker in styles:
            values=[next(r[key] for r in summary if r['stress']==s['name'] and r['method']==method) for s in plan['stress']];ax.plot(range(4),values,marker=marker,color=color,label=label,lw=1.7)
        ax.set_xticks(range(4),['Exact','Equal noise','Mixed quality','Common bias'],rotation=15);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('Fraction / F1');ax.grid(alpha=.2)
    axes[-1].legend(loc='lower left',fontsize=9);fig.suptitle('128 saved RGB-D base scenes x 4 coherent historical-pose stress conditions\nThree methods share declared covariance; fixed change threshold; common global gauge evaluated separately.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary,nonzero_intervals=[r for r in intervals if r['mean_difference']!=0],checks=checks),indent=2))

if __name__=='__main__':main()
