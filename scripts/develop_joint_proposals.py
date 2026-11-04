import hashlib,json,time
from pathlib import Path
import numpy as np
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v4 import solve,coverage

def main():
    config=Path('configs/joint_proposals_v4.json');sampler=json.loads(config.read_text());cfg=json.loads(Path(sampler['base_config']).read_text());base=Path('results/joint_estimation_v1');out=Path('results/joint_proposals_v4_development');out.mkdir(exist_ok=False)
    (out/'frozen_config.json').write_bytes(config.read_bytes());old={r['scene_id']:r for r in [json.loads(l) for l in Path('results/joint_estimation_v2/metrics.jsonl').read_text().splitlines()]};start=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(480):
            a=np.load(base/'scenes'/f'{sid:04d}.npz');obs={k:a[k] for k in a.files if not k.startswith('truth_')};truth={k[6:]:a[k] for k in a.files if k.startswith('truth_')}
            for method in sampler['methods']:
                tick=time.perf_counter();p=solve(obs,cfg['algorithm'],sampler,old[sid]['seed']+400000,method,True)
                row=dict(scene_id=sid,method=method,condition=old[sid]['condition'],seconds=time.perf_counter()-tick,proposal_seconds=p['proposal_seconds'],enumerated=p['enumerated_triples'],**evaluate(p,truth,cfg['evaluation']),**coverage(p,truth,cfg['evaluation']))
                f.write(json.dumps(row,allow_nan=False)+'\n')
            if (sid+1)%120==0:print(json.dumps(dict(scenes=sid+1,elapsed_s=time.perf_counter()-start)),flush=True)
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];summary={}
    for m in sampler['methods']:
        own=[r for r in rows if r['method']==m];hard=[r for r in own if r['scene_id']>=450]
        summary[m]={k:float(np.mean([r[k] for r in own])) for k in ['pose_success','assignment_accuracy','initial_coverage','final_coverage','selection_miss']}
        summary[m].update(f1=float(np.mean([r['f1'] for r in own if r['condition']['change_fraction']>0])),hard_cell_success=sum(r['pose_success'] for r in hard),recovered_old_failures=[r['scene_id'] for r in hard if not old[r['scene_id']]['pose_success'] and r['pose_success']])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    paths=[config,Path(__file__),Path('scripts/joint_estimation_v4.py'),base/'manifest.json',out/'metrics.jsonl']
    (out/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
