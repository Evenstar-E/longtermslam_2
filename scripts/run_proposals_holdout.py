import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.rendered_joint_scene import make_scene
from scripts.joint_estimation import infer,evaluate
from scripts.joint_estimation_v4 import solve,coverage

def predict(obs,alg,sampler,method,seed):
    if method=='robust_sequential':return infer(obs,alg,method,seed)
    config={**alg,'proposal_count':alg['proposal_count']*2} if method=='hard_2x' else alg
    return solve(obs,config,sampler,seed,'hard' if method=='hard_2x' else method,True)

def main():
    config=Path('configs/joint_proposals_holdout_v1.json');plan=json.loads(config.read_text());sampler=json.loads(Path(plan['sampler_config']).read_text());cfg=json.loads(Path(plan['scene_config']).read_text());alg=json.loads(Path(sampler['base_config']).read_text());out=Path('results/joint_proposals_holdout_v1');out.mkdir(exist_ok=False);(out/'scenes').mkdir();(out/'frozen_plan.json').write_bytes(config.read_bytes())
    fields=list(cfg['factors']);conditions=[dict(zip(fields,v)) for v in itertools.product(*(cfg['factors'][k] for k in fields))];metadata=[];begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for cell,condition in enumerate(conditions):
            for k in range(plan['seeds_per_condition']):
                sid=cell*plan['seeds_per_condition']+k;seed=plan['seed_base']+sid;folder=out/'scenes'/f'{sid:04d}';folder.mkdir()
                obs,truth,frames,info,K=make_scene(cfg,alg,condition,seed);valid=info['history_complete'] and 3<=info['detections']<=24
                np.savez_compressed(folder/'measurements.npz',**obs,**{'truth_'+key:v for key,v in truth.items()},K=K)
                for j,frame in enumerate(frames):
                    Image.fromarray(frame['rgb']).save(folder/f'{j}_rgb.png');np.savez_compressed(folder/f'{j}_geometry.npz',depth=frame['depth'],evaluator_ids=frame['ids'],T=frame['T'])
                metadata.append(dict(scene_id=sid,seed=seed,condition=condition,valid=valid,**info))
                for method in plan['methods']:
                    row=dict(scene_id=sid,seed=seed,method=method,condition=condition,valid=valid)
                    if valid:
                        tick=time.perf_counter();p=predict(obs,alg['algorithm'],sampler,method,seed+400000)
                        row.update(seconds=time.perf_counter()-tick,**evaluate(p,truth,alg['evaluation']))
                        if method!='robust_sequential':row.update(**coverage(p,truth,alg['evaluation']),proposal_seconds=p['proposal_seconds'],enumerated=p['enumerated_triples'],hypotheses=p['hypotheses'])
                        eligible=set(info['physical_changed_ids'])&set(info['visible_ids']);good=p['changed']&(p['assignment']==truth['assignment']);correct=set(truth['assignment'][good].tolist())
                        row.update(visible_changed_assets=len(eligible),correctly_reported_changed_assets=len(correct&eligible))
                    else:row.update(reason='invalid_history_or_detection_count',pose_success=False,assignment_accuracy=0.,f1=0.)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush()
            print(json.dumps(dict(cell=cell+1,total=16,scenes=len(metadata),elapsed_s=round(time.perf_counter()-begin,2))),flush=True)
    (out/'scenes.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    paths=[config,Path(plan['sampler_config']),Path(plan['scene_config']),Path(sampler['base_config']),Path(__file__),Path('scripts/joint_estimation_v4.py'),Path('scripts/joint_estimation.py'),Path('scripts/rendered_joint_scene.py'),Path('visual_reinspection/renderer.py'),out/'metrics.jsonl',out/'scenes.json',*sorted((out/'scenes').rglob('*.*'))]
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scenes=len(metadata),records=len(metadata)*len(plan['methods']),seconds=time.perf_counter()-begin),indent=2),encoding='utf-8')

if __name__=='__main__':main()
