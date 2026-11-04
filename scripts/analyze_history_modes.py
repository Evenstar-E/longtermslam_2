"""Independent supplementary reference-qualified task scoring and paired analysis."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.history_modes_v6 import infer,clusters
from scripts.joint_estimation import evaluate

def main():
    out=Path('results/history_modes_v1');plan=json.loads((out/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];index={(r['base_scene'],r['stress'],r['method']):r for r in rows};checks=0
    def check(v):
        nonlocal checks
        assert v;checks+=1
    for path,digest in manifest['files'].items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest)
    check(len(rows)==3072);check(len(index)==len(rows));ids=sorted({r['base_scene'] for r in rows});qualified=[];begin=time.perf_counter()
    # Replaying all cases is needed for a new outcome using per-observation reference validity.
    for level,stress in enumerate(plan['stresses']):
        for sid in ids:
            a=np.load(out/'cases'/f'{level}_{sid:04d}.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k:a['truth_'+k] for k in ['T','assignment','changed']}
            check(np.array_equal(a['truth_provenance'][0],np.arange(24)))
            for method in plan['methods']:
                pred=infer(obs,cfg['algorithm'],sampler,plan['modes'],method,plan['seed_base']+sid+400000);e=evaluate(pred,truth,cfg['evaluation'])
                for k,v in e.items():check(v==index[sid,stress['name'],method][k])
                valid=np.linalg.norm(pred['reference_points']-a['truth_reference'][pred['assignment']],axis=1)<=cfg['algorithm']['change_threshold_m'];same=pred['assignment']==truth['assignment'];good=same&valid;positive=pred['changed'];actual=truth['changed'];tp=int((positive&actual&good).sum());fp=int((positive&(~actual|~good)).sum());fn=int((actual&(~positive|~good)).sum())
                qualified.append(dict(base_scene=sid,stress=stress['name'],method=method,reference_valid_fraction=float(valid.mean()),qualified_tp=tp,qualified_fp=fp,qualified_fn=fn,qualified_f1=2*tp/max(2*tp+fp+fn,1)))
                check(float(valid.mean())==index[sid,stress['name'],method]['reference_valid_fraction'])
        print(json.dumps(dict(replayed=stress['name'],elapsed_s=round(time.perf_counter()-begin,2))),flush=True)
    (out/'reference_qualified.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in qualified),encoding='utf-8');qindex={(r['base_scene'],r['stress'],r['method']):r for r in qualified};summary=[]
    def changed(sid):return ((sid//16)//4)%2==1
    for stress in plan['stresses']:
        for m in plan['methods']:
            rs=[index[s,stress['name'],m] for s in ids];cs=[r for r in rs if changed(r['base_scene'])];qs=[qindex[r['base_scene'],stress['name'],m] for r in cs]
            summary.append(dict(stress=stress['name'],method=m,cases=len(rs),pose_success=float(np.mean([r['pose_success'] for r in rs])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in rs])),change_f1=float(np.mean([r['f1'] for r in cs])),qualified_change_f1=float(np.mean([r['qualified_f1'] for r in qs])),reference_valid_fraction=float(np.mean([r['reference_valid_fraction'] for r in rs])),false_change_rate=sum(r['false_change_on_unchanged'] for r in rs)/sum(r['unchanged_visible'] for r in rs),median_runtime_ms=1000*float(np.median([r['seconds'] for r in rs]))))
    intervals=[];rng=np.random.default_rng(61829)
    for stress in plan['stresses']:
        for control in ['anchor_only','reassociate','anchor_cluster']:
            eligible=np.array([s for s in ids if changed(s)]);groups=[np.flatnonzero(eligible//16==c) for c in sorted(set(eligible//16))]
            d=np.array([qindex[s,stress['name'],'v6_modes']['qualified_f1']-qindex[s,stress['name'],control]['qualified_f1'] for s in eligible]);draws=[float(d[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
            intervals.append(dict(stress=stress['name'],comparison='v6_modes minus '+control,endpoint='reference-qualified change F1',base_scenes=len(eligible),mean_difference=float(d.mean()),percentile95=np.quantile(draws,[.025,.975]).tolist()))
    result=dict(summary=summary,paired_bootstrap=intervals,checks=checks,base_scenes=128,repeated_cases=512,scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='All methods replayed for supplementary reference-qualified F1. Reference validity is distance<=0.25m from original clean history mean, not proof of unique physical identity. Exploratory per-stress paired bootstrap stratified by visual condition.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    styles=[('mean','Mean','#777777','o'),('geomedian','Geometric median','#0072B2','s'),('anchor_only','Reference frame only','#56B4E9','^'),('reassociate','Reassociate history','#E69F00','D'),('anchor_cluster','Reference cluster','#CC79A7','v'),('v6_modes','v6 modes','#009E73','X')]
    fig,axes=plt.subplots(1,3,figsize=(18,5.7),layout='constrained')
    for ax,key,title in zip(axes,['change_f1','qualified_change_f1','reference_valid_fraction'],['Identity-aware change F1','Reference-qualified change F1','Historical reference position validity']):
        for m,label,color,marker in styles:
            vs=[next(r[key] for r in summary if r['stress']==s['name'] and r['method']==m) for s in plan['stresses']];ax.plot(range(4),vs,marker=marker,color=color,label=label,lw=1.5)
        ax.set_xticks(range(4),['Clean','Minority mixing','Majority mixing','Mixing + pose'],rotation=15);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('Fraction / F1');ax.grid(alpha=.2)
    axes[-1].legend(loc='lower left',fontsize=8);fig.suptitle('128 saved RGB-D scenes x 4 historical track stresses; first reference frame validated\nNo discarded predictions; qualified score adds clean-reference geometry check, not physical-identity proof.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    # Fixed scene/corrupted ID diagnostic: choose first changed nominal track from provenance, not method performance.
    a=np.load(out/'cases'/'2_0000.npz');j=int(np.flatnonzero((a['truth_provenance']!=np.arange(24)[None]).any(axis=0))[0]);p=a['history_positions'][:,j];groups=clusters(p,plan['modes']['cluster_radius_m']);fig,ax=plt.subplots(figsize=(7,5),layout='constrained');ax.scatter(p[:,0],p[:,1],s=65,c=np.arange(6),cmap='viridis',label='Historical observations');ax.scatter(p[0,0],p[0,1],marker='*',s=220,c='black',label='Validated reference');ax.scatter(p[:,0].mean(),p[:,1].mean(),marker='x',s=160,c='red',label='Single mean')
    for h,point in enumerate(p):ax.annotate(str(h),(point[0],point[1]),xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='World x (m)',ylabel='World y (m)',title=f'Fixed base scene0 / majority mixing / nominal track{j}\n{len(groups)} spatial modes; labels are historical frame indices');ax.axis('equal');ax.legend();fig.savefig(out/'memory_example.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired_bootstrap=intervals,checks=checks),indent=2))

if __name__=='__main__':main()
