"""All-scene analysis, frontend audit, paired uncertainty and fixed casebook."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scripts.run_rendered_joint import predict
from scripts.joint_estimation import evaluate
from scripts.rendered_joint_scene import extract

def main():
    out=Path('results/rendered_joint_v1');cfg=json.loads((out/'frozen_config.json').read_text());alg=json.loads(Path(cfg['base_algorithm_config']).read_text());manifest=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];scenes=json.loads((out/'scenes.json').read_text());idx={(r['scene_id'],r['method']):r for r in rows};checks=0
    def check(x):
        nonlocal checks
        assert x;checks+=1
    for p,d in manifest['files'].items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d)
    check(len(rows)==128*9);check(len(idx)==len(rows));check(len(scenes)==128)
    summaries=[]
    def aggregate(rs):
        valid=[r for r in rs if r['valid']];change=[r for r in rs if r['condition']['change_fraction']>0];tp=sum(r.get('tp',0) for r in change);fp=sum(r.get('fp',0) for r in change);fn=sum(r.get('fn',0) for r in change)
        return dict(scenes=len(rs),valid_scenes=len(valid),pose_success=float(np.mean([r['pose_success'] for r in rs])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in rs])),change_mean_f1=float(np.mean([r['f1'] for r in change])) if change else None,
           change_micro_f1=2*tp/max(2*tp+fp+fn,1) if change else None,
           visible_asset_change_recall=sum(r.get('correctly_reported_changed_assets',0) for r in change)/max(sum(r.get('visible_changed_assets',0) for r in change),1) if change else None,
           false_change_on_unchanged=sum(r.get('false_change_on_unchanged',0) for r in valid)/max(sum(r.get('unchanged_visible',0) for r in valid),1),
           median_translation_m=float(np.median([r['translation_error_m'] for r in valid])) if valid else None,median_runtime_ms=1000*float(np.median([r['seconds'] for r in valid])) if valid else None)
    for m in cfg['methods']:summaries.append(dict(method=m,**aggregate([r for r in rows if r['method']==m])))
    cells=[]
    for cell in range(16):
        for m in cfg['methods']:
            rs=[idx[s,m] for s in range(cell*8,(cell+1)*8)];cells.append(dict(cell=cell,method=m,**rs[0]['condition'],**aggregate(rs)))
    intervals=[];rng=np.random.default_rng(88437)
    for comparator in ['robust_sequential','joint_v2']:
        for key in ['pose_success','assignment_accuracy','f1']:
            ids=np.array([s for s in range(128) if key!='f1' or idx[s,'joint_v3']['condition']['change_fraction']>0]);d=np.array([float(idx[s,'joint_v3'][key])-float(idx[s,comparator][key]) for s in ids]);groups=[np.flatnonzero(ids//8==c) for c in sorted(set(ids//8))]
            draws=[float(d[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
            intervals.append(dict(comparison='joint_v3 minus '+comparator,endpoint=key,paired_scenes=len(ids),mean_difference=float(d.mean()),percentile95=np.quantile(draws,[.025,.975]).tolist()))
    # Check exact frontend pixel/depth replay and all nine solvers on one scene per cell.
    for sid in range(0,128,8):
        folder=out/'scenes'/f'{sid:04d}';a=np.load(folder/'measurements.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k[6:]:a[k] for k in a.files if k.startswith('truth_')}
        rgb=np.array(Image.open(folder/'6_rgb.png'));geometry=np.load(folder/'6_geometry.npz');detections=extract(rgb,geometry['depth'],a['K'],cfg['render'])
        check(len(detections)==len(obs['current_positions']))
        check(np.array_equal(np.array([d['position'] for d in detections]),obs['current_positions']))
        check(np.array_equal(np.array([d['descriptor'] for d in detections]),obs['current_descriptors']))
        for m in cfg['methods']:
            e=evaluate(predict(obs,alg['algorithm'],m,scenes[sid]['seed']+400000),truth,alg['evaluation'])
            for k,v in e.items():check(v==idx[sid,m][k])
    frontend=dict(total_scenes=128,valid_scenes=sum(s['valid'] for s in scenes),ambiguous_detections=sum(s['ambiguous_detections'] for s in scenes),
       detection_recall_over_visible_assets=sum(s['detected_unique_visible'] for s in scenes)/sum(s['visible_targets'] for s in scenes),
       mean_detections=float(np.mean([s['detections'] for s in scenes])),mean_visible_targets=float(np.mean([s['visible_targets'] for s in scenes])),
       occluded_mean_visible_targets=float(np.mean([s['visible_targets'] for s in scenes if s['condition']['occluder']])),
       unoccluded_mean_visible_targets=float(np.mean([s['visible_targets'] for s in scenes if not s['condition']['occluder']])))
    analysis=dict(summary=summaries,by_condition=cells,frontend=frontend,paired_bootstrap=intervals,checks_passed=checks,scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),uncertainty='Exploratory scene-paired condition-stratified percentile bootstrap;128 synthetic draws from one simple renderer,not external environments;no multiplicity-adjusted claim.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    for name,items in [('summary.csv',summaries),('by_condition.csv',cells)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    names=['Robust sequential','Joint v1','Joint v2','Joint v3','v2 single start','v2 no history','v3 single start','v3 no history','Iterative LS (no weights)'];colors=['#777777','#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7','#E69F00','#CC79A7','#D55E00']
    fig,axes=plt.subplots(1,3,figsize=(18,6.3),layout='constrained')
    for ax,key,title in zip(axes,['pose_success','identity_accuracy','change_mean_f1'],['Pose success <=10 cm AND <=3 deg','Detected-observation identity accuracy','Identity-aware change F1 (change scenes)']):
        values=[s[key] for s in summaries];ax.barh(range(9),values,color=colors);ax.set_yticks(range(9),names);ax.invert_yaxis();ax.set_xlim(0,1.13);ax.set_xlabel('Fraction / F1');ax.set_title(title,fontsize=10)
        for i,v in enumerate(values):ax.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('128 new-layout RGB-D scenes / 896 rendered frames / 9 methods\nImage-derived colour components and depth; annotated history. Simplified planar components.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    # Predefined first scene in four conditions; no algorithm-success-based case selection.
    fig,axes=plt.subplots(4,2,figsize=(12,16),layout='constrained')
    for row,sid in enumerate([0,24,56,120]):
        folder=out/'scenes'/f'{sid:04d}';a=np.load(folder/'measurements.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};pred=predict(obs,alg['algorithm'],'joint_v3',scenes[sid]['seed']+400000)
        for col,frame in enumerate([0,6]):
            ax=axes[row,col];ax.imshow(Image.open(folder/f'{frame}_rgb.png'));ax.axis('off')
            ax.set_title(f"Scene {sid}: {'reference' if col==0 else 'query'}",fontsize=10)
            if col==1:
                for i,rectangle in enumerate(scenes[sid]['rectangles']):
                    x,y,w,h=rectangle;right=pred['assignment'][i]==a['truth_assignment'][i];color='#009E73' if right else '#D55E00';ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,lw=1));ax.text(x,y-2,str(int(pred['assignment'][i])),fontsize=6,color=color)
                ax.set_title(f"Scene {sid}: v3 pose={idx[sid,'joint_v3']['pose_success']}, identity={idx[sid,'joint_v3']['assignment_accuracy']:.2f}",fontsize=10)
    fig.suptitle('Fixed casebook: query boxes from RGB detector; labels are predicted historical IDs\nGreen=correct evaluator identity, orange=wrong. Grey foreground is geometric occluder.');fig.savefig(out/'casebook.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(summary=summaries,frontend=frontend,paired_bootstrap=intervals,checks=checks),indent=2))

if __name__=='__main__':main()
