import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_proposals_holdout import predict
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v4 import coverage
from scripts.rendered_joint_scene import extract

def main():
    out=Path('results/joint_proposals_holdout_v1');plan=json.loads((out/'frozen_plan.json').read_text());sampler=json.loads(Path(plan['sampler_config']).read_text());scene_cfg=json.loads(Path(plan['scene_config']).read_text());alg=json.loads(Path(sampler['base_config']).read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];scenes=json.loads((out/'scenes.json').read_text());index={(r['scene_id'],r['method']):r for r in rows};checks=0
    def check(x):
        nonlocal checks
        assert x;checks+=1
    for p,digest in manifest['files'].items():check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest)
    check(len(rows)==256*6);check(len(index)==len(rows));check(len(scenes)==256)
    def aggregate(rs):
        valid=[r for r in rs if r['valid']];changed=[r for r in rs if r['condition']['change_fraction']>0];joint=rs[0]['method']!='robust_sequential'
        return dict(scenes=len(rs),valid_scenes=len(valid),pose_success=float(np.mean([r['pose_success'] for r in rs])),identity_accuracy=float(np.mean([r['assignment_accuracy'] for r in rs])),change_mean_f1=float(np.mean([r['f1'] for r in changed])) if changed else None,
           initial_coverage=float(np.mean([r.get('initial_coverage',False) for r in rs])) if joint else None,final_coverage=float(np.mean([r.get('final_coverage',False) for r in rs])) if joint else None,
           selection_misses=sum(r.get('selection_miss',False) for r in rs) if joint else None,lost_coverage=sum(r.get('lost_coverage',False) for r in rs) if joint else None,
           median_runtime_ms=1000*float(np.median([r['seconds'] for r in valid])) if valid else None,median_proposal_ms=1000*float(np.median([r['proposal_seconds'] for r in valid])) if joint and valid else None,
           visible_changed_recall=sum(r.get('correctly_reported_changed_assets',0) for r in changed)/max(sum(r.get('visible_changed_assets',0) for r in changed),1) if changed else None)
    summary=[dict(method=m,**aggregate([r for r in rows if r['method']==m])) for m in plan['methods']];cells=[]
    for cell in range(16):
        for m in plan['methods']:
            rs=[index[s,m] for s in range(cell*16,(cell+1)*16)];cells.append(dict(cell=cell,method=m,**rs[0]['condition'],**aggregate(rs)))
    rng=np.random.default_rng(39117);intervals=[]
    for control in ['hard','hard_2x','topk_random']:
        for endpoint in ['pose_success','assignment_accuracy','f1']:
            ids=np.array([s for s in range(256) if endpoint!='f1' or index[s,'geometry']['condition']['change_fraction']>0]);d=np.array([float(index[s,'geometry'][endpoint])-float(index[s,control][endpoint]) for s in ids]);groups=[np.flatnonzero(ids//16==c) for c in sorted(set(ids//16))]
            draws=[float(d[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
            intervals.append(dict(comparison='geometry minus '+control,endpoint=endpoint,paired_scenes=len(ids),mean_difference=float(d.mean()),percentile95=np.quantile(draws,[.025,.975]).tolist()))
    # Inputs from images are re-extracted, then all methods replay once per condition.
    for sid in range(0,256,16):
        folder=out/'scenes'/f'{sid:04d}';a=np.load(folder/'measurements.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k[6:]:a[k] for k in a.files if k.startswith('truth_')}
        geometry=np.load(folder/'6_geometry.npz');d=extract(np.array(Image.open(folder/'6_rgb.png')),geometry['depth'],a['K'],scene_cfg['render'])
        check(np.array_equal(np.array([v['position'] for v in d]),obs['current_positions']));check(np.array_equal(np.array([v['descriptor'] for v in d]),obs['current_descriptors']))
        if not scenes[sid]['valid']:continue
        for m in plan['methods']:
            p=predict(obs,alg['algorithm'],sampler,m,scenes[sid]['seed']+400000);e=evaluate(p,truth,alg['evaluation'])
            if m!='robust_sequential':e.update(coverage(p,truth,alg['evaluation']));check(p['hypotheses']==(49 if m=='hard_2x' else 25))
            for k,v in e.items():check(v==index[sid,m][k])
    matches=[dict(scene_id=s,hard_pose=index[s,'hard']['pose_success'],geometry_pose=index[s,'geometry']['pose_success']) for s in range(256) if index[s,'hard']['pose_success']!=index[s,'geometry']['pose_success']]
    front=dict(valid=sum(s['valid'] for s in scenes),total=256,ambiguous_detections=sum(s['ambiguous_detections'] for s in scenes),visible_detection_recall=sum(s['detected_unique_visible'] for s in scenes)/sum(s['visible_targets'] for s in scenes))
    result=dict(summary=summary,by_condition=cells,paired_bootstrap=intervals,discordant_pose_cases=matches,frontend=front,checks_passed=checks,scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='New seeds/layouts from the same simple renderer;scene-paired exploratory bootstrap,not external-domain or literature superiority. No tuning on holdout outputs.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,rs in [('summary.csv',summary),('by_condition.csv',cells)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    labels=['Hard matching (25)','Hard matching (49)','Top-k random (25)','Geometry v4 (25)','Hybrid (25)','Robust sequential'];colors=['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9']
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
    for ax,key,title in zip(axes,['pose_success','identity_accuracy','change_mean_f1'],['Pose success <=10 cm AND <=3 deg','Detected identity accuracy','Identity-aware change F1 (128 change scenes)']):
        vs=[r[key] for r in summary];ax.barh(range(6),vs,color=colors);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,1.14);ax.set_title(title,fontsize=10);ax.set_xlabel('Fraction / F1')
        for i,v in enumerate(vs):ax.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('256 fresh procedural RGB-D scenes; fixed methods/scales before outcomes\nProposal count shown in labels; geometric enumeration costs extra time. Simple planar renderer.');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(5);w=.24
    for offset,key,label,color in [(-w,'initial_coverage','Initial qualifying pose','#E69F00'),(0,'final_coverage','Any final qualifying pose','#0072B2'),(w,'pose_success','Selected qualifying pose','#009E73')]:
        ax.bar(x+offset,[r[key] for r in summary[:5]],w,label=label,color=color)
    ax.set(xticks=x,xticklabels=labels[:5],ylim=(0,1.08),ylabel='Fraction of all 256 assigned scenes',title='Candidate coverage, convergence and selection measured separately');ax.legend(loc='lower right');fig.savefig(out/'coverage.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(summary=summary,paired_bootstrap=intervals,frontend=front,checks=checks),indent=2))

if __name__=='__main__':main()
