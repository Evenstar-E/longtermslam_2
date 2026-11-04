"""Finite shared-pose/change candidate and simple ray-gating controls."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.run_surface_motion_v2 import save_input
from scripts.partial_surface_scene import make_scene
from scripts.partial_surface_estimation import infer
from scripts.shared_pose_change_v3 import solve
from scripts.joint_estimation import evaluate


def main():
    cfg=json.loads(Path('configs/shared_pose_change_v3.json').read_text());bcfg=json.loads(Path('configs/partial_surface_v1.json').read_text());mcfg=json.loads(Path('configs/surface_motion_v2.json').read_text());mcfg.update(current_max_points=cfg['current_max_points'],surface_samples_per_edge=cfg['surface_samples_per_edge'])
    out=Path('results/shared_pose_change_v3');out.mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Preserve prior run; use a new version')
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');(out/'base_config.json').write_text(json.dumps(bcfg,indent=2),encoding='utf-8');(out/'motion_config.json').write_text(json.dumps(mcfg,indent=2),encoding='utf-8')
    sources=['scripts/shared_pose_change_v3.py','scripts/run_shared_pose_change_v3.py','scripts/surface_motion_v2.py','scripts/partial_surface_scene.py','scripts/partial_surface_estimation.py','scripts/joint_estimation.py','configs/shared_pose_change_v3.json','scripts/run_surface_motion_v2.py','scripts/analyze_partial_surface.py']
    manifest=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},cases=0,records=0,base_predictions={},input_sha256={},exclusions=[])
    oldmanifest=json.loads(Path('results/surface_motion_v2/manifest.json').read_text());oldcfg=json.loads(Path('configs/surface_motion_v2.json').read_text());factors=bcfg['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())];oldids=cfg['diagnostic_old_ids'];total=len(oldids)+8*cfg['fresh_seeds_per_cell'];begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(total):
            if sid<len(oldids):
                oid=oldids[sid];folder=Path('results/partial_surface_v1/cases') if oid<128 else Path('results/surface_motion_v2/cases');source=folder/f'{oid:04d}_input.npz';obs=load_input(source);truth=dict(np.load(folder/f'{oid:04d}_truth.npz'));base=oldmanifest['base_predictions'][str(oid)];cell=oid//16 if oid<128 else (oid-128)//8;seed=(bcfg['seed_start']+oid) if oid<128 else (oldcfg['fresh_seed_start']+oid-128);split='diagnostic_reused'
            else:
                k=sid-len(oldids);cell=k//cfg['fresh_seeds_per_cell'];seed=cfg['seed_start']+k;obs,truth,_=make_scene(bcfg,cells[cell],seed);source=out/'cases'/f'{sid:04d}_input.npz';save_input(source,obs);np.savez_compressed(out/'cases'/f'{sid:04d}_truth.npz',**truth)
                p=infer(obs,bcfg,'surface_joint');base=dict(T=p['T'].tolist(),prediction_assignment=p['assignment'].tolist(),prediction_changed=p['changed'].tolist(),known=p['known'].tolist());split='fresh';oid=None
            manifest['base_predictions'][str(sid)]=dict(base=base,input_path=str(source),old_scene=oid,seed=seed,cell=cell,split=split);manifest['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            for method in cfg['methods']:
                pred=solve(obs,base,bcfg,mcfg,cfg,method,seed)
                # No evaluator labels enter solve; score only after completion.
                ev=evaluate(pred,truth,dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=2));correct=pred['assignment']==truth['assignment'];selected=pred['changed']&truth['changed']&correct
                errors=np.linalg.norm(pred['delta']-truth['delta'],axis=1);motion_success=int((selected&(errors<=.1)).sum())
                row=dict(scene=sid,old_scene=oid,seed=seed,cell=cell,split=split,condition=cells[cell],method=method,**ev,known_count=int(pred['known'].sum()),seconds=pred['seconds'],motion_success=motion_success,correct_positive_motion_errors=errors[selected].tolist(),T=pred['T'].tolist(),delta=pred['delta'].tolist(),prediction_changed=pred['changed'].tolist(),prediction_assignment=pred['assignment'].tolist(),known=pred['known'].tolist(),truth_assignment=truth['assignment'].tolist(),truth_changed=truth['changed'].tolist(),truth_delta=truth['delta'].tolist(),selected_start=pred['selected_start'],hypothesis_scores=pred['scores'],stable_count=pred['stable_count'])
                f.write(json.dumps(row)+'\n');f.flush();manifest['records']+=1
            manifest['cases']+=1
            if (sid+1)%4==0 or sid+1==total:print(f'{sid+1}/{total} cases,{manifest["records"]} records,{time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin;(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Complete',flush=True)


if __name__=='__main__':main()
