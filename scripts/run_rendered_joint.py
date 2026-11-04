"""Render -> anonymous RGB-D detections -> frozen joint inference -> evaluator."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.rendered_joint_scene import make_scene
from scripts.joint_estimation import infer,evaluate
from scripts.joint_estimation_v3 import solve

def predict(obs,cfg,method,seed):
    if method in ['robust_sequential','joint_history','joint_no_change_gate']:return infer(obs,cfg,method,seed)
    variant='v2' if method.startswith('v2_') or method=='joint_v2' else 'v3'
    if method.endswith('_no_history'):obs={**obs,'history_positions':obs['history_positions'][:1],'history_descriptors':obs['history_descriptors'][:1]}
    if method.endswith('_single'):cfg={**cfg,'proposal_count':0}
    return solve(obs,cfg,seed,variant)

def main():
    config=Path('configs/rendered_joint_v1.json');cfg=json.loads(config.read_text());algorithm_cfg=json.loads(Path(cfg['base_algorithm_config']).read_text());out=Path('results/rendered_joint_v1');out.mkdir(exist_ok=False);(out/'scenes').mkdir()
    (out/'frozen_config.json').write_bytes(config.read_bytes());factors=list(cfg['factors']);conditions=[dict(zip(factors,v)) for v in itertools.product(*(cfg['factors'][f] for f in factors))];metadata=[];started=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for cell,condition in enumerate(conditions):
            for k in range(cfg['seeds_per_condition']):
                sid=cell*cfg['seeds_per_condition']+k;seed=cfg['seed_base']+sid;folder=out/'scenes'/f'{sid:04d}';folder.mkdir()
                obs,truth,frames,info,K=make_scene(cfg,algorithm_cfg,condition,seed)
                np.savez_compressed(folder/'measurements.npz',**obs,**{'truth_'+k:v for k,v in truth.items()},K=K)
                for j,frame in enumerate(frames):
                    Image.fromarray(frame['rgb']).save(folder/f'{j}_rgb.png');np.savez_compressed(folder/f'{j}_geometry.npz',depth=frame['depth'],evaluator_ids=frame['ids'],T=frame['T'])
                valid=info['history_complete'] and 3<=info['detections']<=24
                metadata.append(dict(scene_id=sid,seed=seed,condition=condition,valid=valid,**info))
                for method in cfg['methods']:
                    row=dict(scene_id=sid,seed=seed,condition=condition,method=method,valid=valid)
                    if valid:
                        tick=time.perf_counter();pred=predict(obs,algorithm_cfg['algorithm'],method,seed+400000);row.update(seconds=time.perf_counter()-tick,**evaluate(pred,truth,algorithm_cfg['evaluation']))
                        good=pred['changed']&(pred['assignment']==truth['assignment']);reported_ids=set(truth['assignment'][good].tolist());eligible=set(info['physical_changed_ids'])&set(info['visible_ids']);row.update(visible_changed_assets=len(eligible),correctly_reported_changed_assets=len(reported_ids&eligible))
                    else:row.update(reason='incomplete_historical_annotation' if not info['history_complete'] else 'invalid_detection_count',pose_success=False,assignment_accuracy=0.,f1=0.)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush()
            print(json.dumps(dict(cell=cell+1,total=16,scenes=len(metadata),elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    (out/'scenes.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    sources=[config,Path(__file__),Path('scripts/joint_estimation_v3.py'),Path('scripts/rendered_joint_scene.py'),Path('scripts/joint_estimation.py'),Path('visual_reinspection/renderer.py'),Path(cfg['base_algorithm_config'])]
    paths=sources+[out/'metrics.jsonl',out/'scenes.json',*sorted((out/'scenes').rglob('*.*'))]
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scenes=len(metadata),records=len(metadata)*len(cfg['methods']),seconds=time.perf_counter()-started),indent=2),encoding='utf-8')

if __name__=='__main__':main()
