import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import transform,evaluate
from scripts.joint_estimation_v4 import coverage
from scripts.history_modes_v6 import infer

def corrupt(original,stress,seed):
    rng=np.random.default_rng(seed);P=original['history_positions'].copy();F=original['history_descriptors'].copy();H,N,_=P.shape;distance=((F[0,:,None]-F[0,None])**2).sum(axis=2);pairs=sorted((float(distance[i,j]),i,j) for i in range(N) for j in range(i+1,N));used=set();selected=[];target=int(round(N*stress['pair_fraction']/2))
    for _,i,j in pairs:
        if i in used or j in used:continue
        if len(selected)>=target:break
        used.update([i,j]);selected.append((i,j))
    provenance=np.tile(np.arange(N),(H,1))
    for i,j in selected:
        views=rng.choice(np.arange(1,H),stress['swapped_views'],replace=False)
        for h in views:P[h,[i,j]]=P[h,[j,i]];F[h,[i,j]]=F[h,[j,i]];provenance[h,[i,j]]=provenance[h,[j,i]]
    for h in range(1,H):
        E=np.eye(4);E[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(stress['rotation_sigma_deg']),3)).as_matrix();E[:3,3]=rng.normal(0,stress['translation_sigma_m'],3);P[h]=transform(P[h],E)
    return {**original,'history_positions':P,'history_descriptors':F},provenance,selected

def main():
    config=Path('configs/history_modes_v1.json');plan=json.loads(config.read_text());base=Path(plan['source']);cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());out=Path('results/history_modes_v1');out.mkdir(exist_ok=False);(out/'cases').mkdir();(out/'frozen_config.json').write_bytes(config.read_bytes());ids=[c*16+k for c in range(16) for k in range(8)];begin=time.perf_counter();source=[]
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for level,stress in enumerate(plan['stresses']):
            for sid in ids:
                path=base/'scenes'/f'{sid:04d}'/'measurements.npz';source.append(path);data=np.load(path);original={k:data[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k:data['truth_'+k] for k in ['T','assignment','changed']};obs,provenance,pairs=corrupt(original,stress,plan['seed_base']+sid+level*100000)
                np.savez_compressed(out/'cases'/f'{level}_{sid:04d}.npz',**obs,truth_T=truth['T'],truth_assignment=truth['assignment'],truth_changed=truth['changed'],truth_provenance=provenance,truth_reference=original['history_positions'].mean(axis=0))
                for method in plan['methods']:
                    tick=time.perf_counter();pred=infer(obs,cfg['algorithm'],sampler,plan['modes'],method,plan['seed_base']+sid+400000);metrics=evaluate(pred,truth,cfg['evaluation']);good_reference=np.linalg.norm(pred['reference_points']-original['history_positions'].mean(axis=0)[pred['assignment']],axis=1)<=cfg['algorithm']['change_threshold_m'];metrics.update(reference_valid_fraction=float(good_reference.mean()),memory_modes=pred.get('memory_modes',24))
                    f.write(json.dumps(dict(base_scene=sid,stress=stress['name'],method=method,seconds=time.perf_counter()-tick,**metrics,**coverage(pred,truth,cfg['evaluation'])),allow_nan=False)+'\n')
                f.flush()
            print(json.dumps(dict(stress=stress['name'],cases=(level+1)*128,elapsed_s=round(time.perf_counter()-begin,2))),flush=True)
    paths=[config,Path(plan['base_algorithm']),Path(plan['sampler']),Path(__file__),Path('scripts/history_modes_v6.py'),Path('scripts/joint_estimation.py'),Path('scripts/joint_estimation_v4.py'),out/'metrics.jsonl',*sorted(set(source)),*sorted((out/'cases').glob('*.npz'))]
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},base_scenes=128,cases=512,records=3072,seconds=time.perf_counter()-begin),indent=2),encoding='utf-8')

if __name__=='__main__':main()
