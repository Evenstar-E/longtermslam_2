import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import evaluate,transform
from scripts.joint_estimation_v4 import solve as solve_v4,coverage
from scripts.joint_estimation_v5 import solve as solve_v5,fuse_history,skew

def perturb(original,stress,measurement_sigma,seed):
    rng=np.random.default_rng(seed);common=np.eye(4);common[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(stress['common_rotation_sigma_deg']),3)).as_matrix();common[:3,3]=rng.normal(0,stress['common_translation_sigma_m'],3)
    positions=[];covariances=[]
    for h,p in enumerate(original['history_positions']):
        t=stress['translation_sigma_m'][h];r=np.deg2rad(stress['rotation_sigma_deg'][h]);E=np.eye(4);E[:3,:3]=Rotation.from_rotvec(rng.normal(0,r,3)).as_matrix();E[:3,3]=rng.normal(0,t,3)
        observed=transform(transform(p,E),common);C=[]
        for x in observed:
            J=-skew(x);C.append(t*t*np.eye(3)+r*r*(J@J.T)+measurement_sigma**2*np.eye(3))
        positions.append(observed);covariances.append(C)
    obs={**original,'history_positions':np.array(positions),'history_covariances':np.array(covariances),'current_variance_m2':measurement_sigma**2}
    return obs,common

def predict(obs,cfg,sampler,method,seed):
    if method=='v5_uncertainty':return solve_v5(obs,cfg,sampler,seed,True)
    if method=='weighted_memory':
        reference,_=fuse_history(obs['history_positions'],obs['history_covariances']);obs={**obs,'history_positions':reference[None]}
    return solve_v4(obs,cfg,sampler,seed,'geometry',True)

def main():
    config=Path('configs/history_uncertainty_v1.json');plan=json.loads(config.read_text());base=Path(plan['source']);cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());out=Path('results/history_uncertainty_v1');out.mkdir(exist_ok=False);(out/'cases').mkdir();(out/'frozen_config.json').write_bytes(config.read_bytes())
    ids=[cell*16+k for cell in range(16) for k in range(8)];source_paths=[];begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for level,stress in enumerate(plan['stress']):
            for sid in ids:
                path=base/'scenes'/f'{sid:04d}'/'measurements.npz';a=np.load(path);source_paths.append(path);original={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};truth={k[6:]:a[k] for k in a.files if k.startswith('truth_')}
                obs,common=perturb(original,stress,plan['measurement_sigma_m'],plan['seed_base']+sid+100000*level);relative_truth={**truth,'T':common@truth['T']}
                np.savez_compressed(out/'cases'/f'{level}_{sid:04d}.npz',**obs,**{'truth_'+k:v for k,v in truth.items()},truth_common=common)
                for method in plan['methods']:
                    tick=time.perf_counter();p=predict(obs,cfg['algorithm'],sampler,method,plan['seed_base']+sid+400000)
                    e=evaluate(p,truth,cfg['evaluation']);rel=evaluate(p,relative_truth,cfg['evaluation']);cover=coverage(p,relative_truth,cfg['evaluation'])
                    f.write(json.dumps(dict(base_scene=sid,stress=stress['name'],method=method,seconds=time.perf_counter()-tick,relative_pose_success=rel['pose_success'],relative_translation_m=rel['translation_error_m'],**e,**cover),allow_nan=False)+'\n')
                f.flush()
            print(json.dumps(dict(stress=stress['name'],cases=(level+1)*128,elapsed_s=round(time.perf_counter()-begin,2))),flush=True)
    paths=[config,Path(plan['base_algorithm']),Path(plan['sampler']),Path(__file__),Path('scripts/joint_estimation_v5.py'),Path('scripts/joint_estimation_v4.py'),out/'metrics.jsonl',*sorted(set(source_paths)),*sorted((out/'cases').glob('*.npz'))]
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},base_scenes=128,cases=512,records=1536,seconds=time.perf_counter()-begin),indent=2),encoding='utf-8')

if __name__=='__main__':main()
