"""Explicit engineering recovery: retain prefix, adapt fragments, append remainder.

The original runner wrote its manifest only at completion; prefix manifest is
therefore reconstructed with exact input/prediction hashes and verified by replay.
"""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.run_surface_motion_v2 import save_input
from scripts.partial_surface_scene import make_scene
from scripts.partial_surface_estimation import infer
from scripts.shared_pose_change_v3 import solve
from scripts.fragment_adapter_v3 import merge_excess_fragments
from scripts.joint_estimation import evaluate


def main():
    out=Path('results/shared_pose_change_v3');cfg=json.loads((out/'frozen_config.json').read_text());bcfg=json.loads((out/'base_config.json').read_text());mcfg=json.loads((out/'motion_config.json').read_text());metrics=out/'metrics.jsonl';prefix=metrics.read_bytes();rows=[json.loads(s) for s in prefix.decode().splitlines()]
    assert len(rows)==256 and rows[-1]['scene']==63 and rows[-1]['method']=='joint_profile'
    oldids=cfg['diagnostic_old_ids'];factors=bcfg['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    sources=['scripts/shared_pose_change_v3.py','scripts/run_shared_pose_change_v3.py','scripts/surface_motion_v2.py','scripts/partial_surface_scene.py','scripts/partial_surface_estimation.py','scripts/joint_estimation.py','configs/shared_pose_change_v3.json','scripts/run_surface_motion_v2.py','scripts/analyze_partial_surface.py','scripts/fragment_adapter_v3.py','scripts/resume_shared_pose_change_v3.py']
    manifest=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},cases=64,records=256,base_predictions={},input_sha256={},exclusions=[],recovery=dict(reason='Excess detector fragments exceeded Hungarian target count at scene64',preserved_prefix_bytes=len(prefix),preserved_prefix_sha256=hashlib.sha256(prefix).hexdigest(),prefix_hashes_reconstructed_after_crash=True,policy='Only when query count exceeds reference count: merge same-appearance fragments assigned nearest observed reference under supplied initial pose; same adapted inputs for all four methods',adapted_cases=[]))
    for r in rows:
        if r['method']!='frozen_v1':continue
        sid=r['scene'];oid=r['old_scene']
        source=(Path('results/partial_surface_v1/cases') if oid<128 else Path('results/surface_motion_v2/cases'))/f'{oid:04d}_input.npz' if oid is not None else out/'cases'/f'{sid:04d}_input.npz'
        base=dict(T=r['T'],prediction_assignment=r['prediction_assignment'],prediction_changed=r['prediction_changed'],known=r['known'])
        manifest['base_predictions'][str(sid)]=dict(base=base,input_path=str(source),old_scene=oid,seed=r['seed'],cell=r['cell'],split=r['split']);manifest['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'recovery_protocol.json').write_text(json.dumps(manifest['recovery'],indent=2),encoding='utf-8')
    begin=time.perf_counter()
    with metrics.open('a',encoding='utf-8') as f:
        for sid in range(64,81):
            k=sid-len(oldids);cell=k//8;seed=cfg['seed_start']+k;raw=out/'cases'/f'{sid:04d}_input.npz';tp=out/'cases'/f'{sid:04d}_truth.npz'
            if raw.exists():obs=load_input(raw);truth=dict(np.load(tp))
            else:
                obs,truth,_=make_scene(bcfg,cells[cell],seed);save_input(raw,obs);np.savez_compressed(tp,**truth)
            obs,groups=merge_excess_fragments(obs,bcfg['points_per_object']);source=raw
            if any(len(g)>1 for g in groups):
                original=truth;ids=[];conflicts=0
                for g in groups:
                    labels=original['assignment'][g];conflicts+=int(len(set(labels.tolist()))>1);ids.append(g[0])
                truth=dict(original);truth['assignment']=original['assignment'][ids];truth['changed']=original['changed'][ids];truth['delta']=original['delta'][ids]
                # This batch's adapted groups must have pure evaluator ownership;
                # if not, stop rather than silently dropping a distinct asset.
                assert conflicts==0,'Mixed-asset merge requires explicit unmatched evaluation'
                source=out/'cases'/f'{sid:04d}_adapter_input.npz';save_input(source,obs);np.savez_compressed(out/'cases'/f'{sid:04d}_adapter_truth.npz',**truth)
                manifest['recovery']['adapted_cases'].append(dict(scene=sid,raw_detections=sum(map(len,groups)),adapted_detections=len(groups),groups=groups,evaluator_mixed_asset_groups=conflicts))
            p=infer(obs,bcfg,'surface_joint');base=dict(T=p['T'].tolist(),prediction_assignment=p['assignment'].tolist(),prediction_changed=p['changed'].tolist(),known=p['known'].tolist())
            manifest['base_predictions'][str(sid)]=dict(base=base,input_path=str(source),old_scene=None,seed=seed,cell=cell,split='fresh');manifest['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            for method in cfg['methods']:
                pred=solve(obs,base,bcfg,mcfg,cfg,method,seed);ev=evaluate(pred,truth,dict(pose_translation_tolerance_m=.1,pose_rotation_tolerance_deg=2));correct=pred['assignment']==truth['assignment'];selected=pred['changed']&truth['changed']&correct;errors=np.linalg.norm(pred['delta']-truth['delta'],axis=1)
                row=dict(scene=sid,old_scene=None,seed=seed,cell=cell,split='fresh',condition=cells[cell],method=method,**ev,known_count=int(pred['known'].sum()),seconds=pred['seconds'],motion_success=int((selected&(errors<=.1)).sum()),correct_positive_motion_errors=errors[selected].tolist(),T=pred['T'].tolist(),delta=pred['delta'].tolist(),prediction_changed=pred['changed'].tolist(),prediction_assignment=pred['assignment'].tolist(),known=pred['known'].tolist(),truth_assignment=truth['assignment'].tolist(),truth_changed=truth['changed'].tolist(),truth_delta=truth['delta'].tolist(),selected_start=pred['selected_start'],hypothesis_scores=pred['scores'],stable_count=pred['stable_count'])
                f.write(json.dumps(row)+'\n');f.flush();manifest['records']+=1
            manifest['cases']+=1;(out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(f'{sid+1}/81 cases; recovery elapsed {time.perf_counter()-begin:.1f}s',flush=True)
    assert metrics.read_bytes()[:len(prefix)]==prefix
    manifest['recovery_seconds']=time.perf_counter()-begin;manifest['interrupted_run_completed64_time_s']=546.9
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');(out/'recovery_protocol.json').write_text(json.dumps(manifest['recovery'],indent=2),encoding='utf-8')


if __name__=='__main__':main()
