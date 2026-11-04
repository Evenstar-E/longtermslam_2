"""Repair under-spent single-object restart control, preserving original outputs.

Original extra-restart loop stopped if a random all-null start used zero ICP.
Only fill its missing call budget; never call this a new algorithm or a hidden rerun.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.run_rigid_identity_cut_ablation_v1 import load_problem
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem

cfg=read('configs/rigid_identity_bounds_v1.json');out=Path('results/rigid_identity_budget_fill_v1');out.mkdir(exist_ok=False)
save(out/'start_manifest.json',{'source_sha256':sha(__file__),'reason':'Posthoc execution repair: all-null random start made old loop stop below intended ICP budget in three singleton classes. Original outputs retained. No candidate changes.'})
for si,scene in enumerate(cfg['spaces']):
    dest=out/scene;dest.mkdir()
    for folder in sorted((Path('results/rigid_identity_cached_v1')/scene).glob('class_*')):
        result=read(folder/'results.json');target=result['bounded_extra_icp_calls'];used=result['ordinary_extra_icp_calls'];missing=max(0,target-used);best=result['controls']['ordinary_extra_restarts'];calls=0
        if missing:
            H,C,shape=load_problem(scene,int(folder.name.split('_')[1]),cfg);p=CachedJointProblem(H,C,shape,cfg);rng=np.random.default_rng(cfg['seed']+si*1000+int(folder.name.split('_')[1]));assert p.m==p.n==1
            while calls<missing:
                T=np.array(best['poses'][0]);T[:3,:3]=Rotation.from_rotvec(rng.normal(0,.05,3)).as_matrix()@T[:3,:3];T[:3,3]+=rng.normal(0,.02,3)
                trial=p.refine(np.array([0]),[T],1);calls=p.fit_calls
                if trial['energy']<best['energy']:best=trial
        save(dest/(folder.name+'.json'),{'best':best,'original_extra_calls':used,'supplemental_calls':calls,'total_extra_calls':used+calls,'candidate_extra_calls':target,'original_result_sha256':sha(folder/'results.json')})
save(out/'seal.json',{'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*.json') if p.name!='seal.json'},'candidate_unchanged':True})
print('FILLED missing ordinary-control budget; original predictions untouched')
