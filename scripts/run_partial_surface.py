"""Frozen procedural development batch; serialize predictions before scoring."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scripts.partial_surface_scene import make_scene
from scripts.partial_surface_estimation import infer
from scripts.joint_estimation_v4 import solve
from scripts.joint_estimation import evaluate


def main():
    cfg=json.loads(Path('configs/partial_surface_v1.json').read_text());out=Path('results/partial_surface_v1')
    out.mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True);(out/'visuals').mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Preserve existing result; use a new version for reruns')
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    alg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text())['algorithm'];sampler=json.loads(Path('configs/joint_proposals_v4.json').read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('configs/partial_surface_v1.json'),Path('scripts/partial_surface_scene.py'),Path('scripts/partial_surface_estimation.py'),Path(__file__),Path('scripts/joint_estimation_v4.py'),Path('scripts/joint_estimation.py')]}
    manifest=dict(source_sha256=hashes,oracle='oracle_pose_surface receives true pose; all other inference receives obs only',exclusions=[],cases=0,records=0)
    factors=cfg['factors'];keys=list(factors);begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for cell,values in enumerate(itertools.product(*(factors[k] for k in keys))):
            condition=dict(zip(keys,values))
            for repeat in range(cfg['seeds_per_cell']):
                sid=cell*cfg['seeds_per_cell']+repeat;seed=cfg['seed_start']+sid
                obs,truth,visuals=make_scene(cfg,condition,seed)
                if len(obs['query'])>len(obs['bounds']):raise RuntimeError(f'Too many detector fragments in scene {sid}')
                lengths=np.array([len(q['points']) for q in obs['query']]);points=np.concatenate([q['points'] for q in obs['query']])
                np.savez_compressed(out/'cases'/f'{sid:04d}_input.npz',bounds=obs['bounds'],face_support=obs['face_support'],reference_centroids=obs['reference_centroids'],reference_features=obs['reference_features'],points=points,lengths=lengths,query_features=np.array([q['feature'] for q in obs['query']]),query_pixels=np.concatenate([q['all_pixels'] for q in obs['query']]),pixel_lengths=np.array([len(q['all_pixels']) for q in obs['query']]),initial_T=obs['initial_T'],depth=obs['depth'],rgb=obs['rgb'],rays=obs['rays'])
                np.savez_compressed(out/'cases'/f'{sid:04d}_truth.npz',**truth)
                for method in cfg['methods']:
                    if method=='centroid_v4':
                        v4obs=dict(history_positions=obs['reference_centroids'][None],history_descriptors=obs['reference_features'][None],current_positions=np.array([q['points'].mean(0) for q in obs['query']]),current_descriptors=np.array([q['feature'] for q in obs['query']]),initial_T=obs['initial_T'])
                        start=time.perf_counter();pred=solve(v4obs,alg,sampler,seed);pred['seconds']=time.perf_counter()-start;pred['known']=np.ones(len(lengths),bool)
                    else:pred=infer(obs,cfg,method,truth['T'] if method=='oracle_pose_surface' else None)
                    ev=evaluate(pred,truth,dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=2))
                    row=dict(scene=sid,seed=seed,cell=cell,condition=condition,method=method,seconds=pred['seconds'],known_count=int(pred['known'].sum()),**ev)
                    row.update(residual_m=pred['residual_m'].tolist(),known=pred['known'].tolist(),prediction_changed=pred['changed'].tolist(),prediction_assignment=pred['assignment'].tolist(),truth_assignment=truth['assignment'].tolist(),truth_changed=truth['changed'].tolist(),T=pred['T'].tolist())
                    if 'delta' in pred:row['estimated_delta']=pred['delta'].tolist()
                    f.write(json.dumps(row)+'\n');f.flush();manifest['records']+=1
                if repeat==0:
                    from PIL import Image
                    Image.fromarray(obs['rgb']).save(out/'visuals'/f'cell_{cell}_query.png')
                    for h,(rgb,_,_) in enumerate(visuals['history']):Image.fromarray(rgb).save(out/'visuals'/f'cell_{cell}_history_{h}.png')
                manifest['cases']+=1
            print(f'cell {cell+1}/8 done; {manifest["cases"]} cases, {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
