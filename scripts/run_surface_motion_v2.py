"""Run motion-only branches with fixed v1 poses/assignments/change decisions."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.partial_surface_scene import make_scene
from scripts.partial_surface_estimation import infer,signed_box_distance
from scripts.joint_estimation import transform
from scripts.surface_motion_v2 import stable_pose_information,fit_motion


def save_input(path,obs):
    arrays={k:obs[k] for k in ('bounds','face_support','reference_centroids','reference_features','initial_T','depth','rgb','rays')}
    arrays.update(points=np.concatenate([q['points'] for q in obs['query']]),lengths=np.array([len(q['points']) for q in obs['query']]),query_features=np.array([q['feature'] for q in obs['query']]),query_pixels=np.concatenate([q['all_pixels'] for q in obs['query']]),pixel_lengths=np.array([len(q['all_pixels']) for q in obs['query']]))
    np.savez_compressed(path,**arrays)


def main():
    cfg=json.loads(Path('configs/surface_motion_v2.json').read_text());basecfg=json.loads(Path(cfg['source_config']).read_text());out=Path('results/surface_motion_v2');out.mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Preserve existing predictions; create a new version')
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');(out/'base_config.json').write_text(json.dumps(basecfg,indent=2),encoding='utf-8')
    sources=['configs/surface_motion_v2.json','configs/partial_surface_v1.json','scripts/surface_motion_v2.py','scripts/run_surface_motion_v2.py','scripts/partial_surface_scene.py','scripts/partial_surface_estimation.py','scripts/analyze_partial_surface.py','scripts/joint_estimation.py']
    manifest=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},cases=0,records=0,exclusions=[],base_predictions={},input_sha256={})
    old=[json.loads(s) for s in Path('results/partial_surface_v1/metrics.jsonl').read_text().splitlines()];old={r['scene']:r for r in old if r['method']=='surface_joint'}
    factors=basecfg['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())];begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(192):
            fresh=sid>=128
            if fresh:
                cell=(sid-128)//cfg['fresh_seeds_per_cell'];seed=cfg['fresh_seed_start']+sid-128;obs,truth,_=make_scene(basecfg,cells[cell],seed)
                save_input(out/'cases'/f'{sid:04d}_input.npz',obs);np.savez_compressed(out/'cases'/f'{sid:04d}_truth.npz',**truth)
                pred=infer(obs,basecfg,'surface_joint');T=pred['T'];A=pred['assignment'];pc=pred['changed'];known=pred['known']
                base=dict(T=T.tolist(),prediction_assignment=A.tolist(),prediction_changed=pc.tolist(),known=known.tolist(),residual_m=pred['residual_m'].tolist(),estimated_delta=pred['delta'].tolist())
                source=out/'cases'/f'{sid:04d}_input.npz'
            else:
                cell=sid//16;seed=basecfg['seed_start']+sid;source=Path('results/partial_surface_v1/cases')/f'{sid:04d}_input.npz';obs=load_input(source);truth=dict(np.load(Path('results/partial_surface_v1/cases')/f'{sid:04d}_truth.npz'));base=old[sid]
                T=np.array(base['T']);A=np.array(base['prediction_assignment']);pc=np.array(base['prediction_changed']);known=np.array(base['known'])
            manifest['base_predictions'][str(sid)]=base;manifest['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            r=np.array([np.median(np.abs(signed_box_distance(transform(q['points'],T),obs['bounds'][j]))) for q,j in zip(obs['query'],A)]);weights=1/(1+(r/basecfg['group_scale_m'])**4)
            pose_information=stable_pose_information(obs,T,A,weights)
            for method in cfg['methods']:
                start=time.perf_counter();fits={}
                for i in np.flatnonzero(pc&known):fits[str(int(i))]=fit_motion(obs,T,A,int(i),cfg,method,pose_information)
                # Predictions finalized before reading evaluator labels below.
                row=dict(scene=sid,split='fresh' if fresh else 'reused',seed=seed,cell=cell,condition=cells[cell],method=method,fits=fits,seconds=time.perf_counter()-start,pose_information=pose_information.tolist(),base_changed=pc.tolist(),base_known=known.tolist(),base_assignment=A.tolist())
                row.update(truth_assignment=truth['assignment'].tolist(),truth_changed=truth['changed'].tolist(),truth_delta=truth['delta'].tolist())
                f.write(json.dumps(row)+'\n');f.flush();manifest['records']+=1
            manifest['cases']+=1
            if (sid+1)%16==0:print(f'{sid+1}/192 cases, {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin;(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(f'Complete: {manifest["records"]} records',flush=True)


if __name__=='__main__':main()
