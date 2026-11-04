import hashlib,json,time
from pathlib import Path
import numpy as np
from scripts.history_alignment_control import infer
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v4 import coverage

def main():
    base=Path('results/history_modes_v1');plan=json.loads((base/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());out=Path('results/history_alignment_control_v1');out.mkdir(exist_ok=False);ids=[c*16+k for c in range(16) for k in range(8)];start=time.perf_counter()
    (out/'protocol.json').write_text(json.dumps(dict(stage='Post-v6 diagnostic conventional control; reused cases,not independent confirmation',source=str(base),change='Realign historical views to validated first frame using v4 before averaging; no current data during repair',budget='Five extra historical registrations plus current registration,each25starts8iterations',methods_compared='all original baselines retained'),indent=2),encoding='utf-8')
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for level,stress in enumerate(plan['stresses']):
            for sid in ids:
                a=np.load(base/'cases'/f'{level}_{sid:04d}.npz');obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k:a['truth_'+k] for k in ['T','assignment','changed']};tick=time.perf_counter();pred=infer(obs,cfg['algorithm'],sampler,plan['seed_base']+sid+400000);elapsed=time.perf_counter()-tick;e=evaluate(pred,truth,cfg['evaluation']);valid=np.linalg.norm(pred['reference_points']-a['truth_reference'][pred['assignment']],axis=1)<=cfg['algorithm']['change_threshold_m'];good=valid&(pred['assignment']==truth['assignment']);p=pred['changed'];c=truth['changed'];tp=int((p&c&good).sum());fp=int((p&(~c|~good)).sum());fn=int((c&(~p|~good)).sum())
                f.write(json.dumps(dict(base_scene=sid,stress=stress['name'],method='history_alignment',seconds=elapsed,reference_valid_fraction=float(valid.mean()),qualified_f1=2*tp/max(2*tp+fp+fn,1),qualified_tp=tp,qualified_fp=fp,qualified_fn=fn,**e,**coverage(pred,truth,cfg['evaluation'])),allow_nan=False)+'\n');f.flush()
            print(json.dumps(dict(stress=stress['name'],elapsed_s=round(time.perf_counter()-start,2))),flush=True)
    paths=[Path(__file__),Path('scripts/history_alignment_control.py'),Path('scripts/joint_estimation_v4.py'),base/'manifest.json',out/'metrics.jsonl',out/'protocol.json'];(out/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')

if __name__=='__main__':main()
