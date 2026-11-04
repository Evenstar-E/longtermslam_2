import hashlib,json
from pathlib import Path
import numpy as np
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v3 import solve

def main():
    base=Path('results/joint_estimation_v1');cfg=json.loads((base/'frozen_config.json').read_text());out=Path('results/joint_search_v3');out.mkdir(exist_ok=False)
    old={r['scene_id']:r for r in [json.loads(l) for l in Path('results/joint_estimation_v2/metrics.jsonl').read_text().splitlines()]}
    rows=[];diagnosis=[]
    for sid in range(480):
        scene=np.load(base/'scenes'/f'{sid:04d}.npz');obs={k:scene[k] for k in scene.files if not k.startswith('truth_')};truth={k[6:]:scene[k] for k in scene.files if k.startswith('truth_')}
        for variant in ['v3','v2','single','no_history']:
            pred=solve(obs,cfg['algorithm'],old[sid]['seed']+400000,variant,diagnostics=variant=='v2')
            metrics=evaluate(pred,truth,cfg['evaluation'])
            rows.append(dict(scene_id=sid,method=variant,condition=old[sid]['condition'],**metrics))
            if variant=='v2':
                for k,v in metrics.items():assert v==old[sid][k],(sid,k)
                if sid>=450:
                    successes=[]
                    for i,s in enumerate(pred['solutions']):
                        e=evaluate(dict(**{k:s[k] for k in ['T','assignment']},changed=s['residual_m']>.25),truth,cfg['evaluation'])
                        if e['pose_success']:successes.append(i)
                    robust_idx=min(range(len(pred['solutions'])),key=lambda i:pred['solutions'][i]['robust_score'])
                    diagnosis.append(dict(scene_id=sid,v2_pose_success=metrics['pose_success'],any_successful_candidate=bool(successes),v2_selected=pred['selected_index'],robust_selected=robust_idx,robust_selected_success=robust_idx in successes,
                        candidates=[dict(full_score=s['full_score'],robust_score=s['robust_score'],pose_success=i in successes) for i,s in enumerate(pred['solutions'])]))
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (out/'hard_cell_diagnosis.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
    summary={m:dict(pose_success=float(np.mean([r['pose_success'] for r in rows if r['method']==m])),identity=float(np.mean([r['assignment_accuracy'] for r in rows if r['method']==m])),f1=float(np.mean([r['f1'] for r in rows if r['method']==m and r['condition']['change_fraction']>0]))) for m in ['v3','v2','single','no_history']}
    summary['hard_cell']=dict(scenes=30,v2_failures=sum(not d['v2_pose_success'] for d in diagnosis),failures_with_successful_candidate=sum(not d['v2_pose_success'] and d['any_successful_candidate'] for d in diagnosis),failures_recovered_by_robust_ranking=sum(not d['v2_pose_success'] and d['robust_selected_success'] for d in diagnosis))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    paths=[Path(__file__),Path('scripts/joint_estimation_v3.py'),base/'manifest.json',out/'metrics.jsonl',out/'hard_cell_diagnosis.json']
    (out/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
