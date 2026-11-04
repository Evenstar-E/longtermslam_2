"""Numerical equivalence on actual anonymous observed patches, no evaluator IDs."""
import copy,time
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save
from scripts.run_rigid_identity_cut_ablation_v1 import load_problem
from inspection_decisions.rigid_identity_bounds_v1 import JointProblem,pose
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem

cfg=read('configs/rigid_identity_bounds_v1.json');H,C,shape=load_problem('scene_c',5,cfg)
# Software check uses observed subset, NOT added research samples.
H=copy.deepcopy(H[:2]);C=copy.deepcopy(C[:2]);shape=shape[:2,:2]
for v in H+C:v['xyz']=v['xyz'][::max(1,len(v['xyz'])//350)];v['rgb']=v['rgb'][::max(1,len(v['rgb'])//350)]
a=JointProblem(H,C,shape,cfg);b=CachedJointProblem(H,C,shape,cfg);rng=np.random.default_rng(42);checks=[]
for k in range(12):
    poses=[pose(rng.uniform(-.1,.1,3),rng.uniform(a.position_lo,a.position_hi),a.anchor[i]) for i in range(a.m)]
    ca=a.costs(poses);cb=b.costs(poses);checks.append(bool(np.array_equal(ca,cb)))
    for assignment in [np.array([0,1]),np.array([0,0]),np.array([-1,1])]:checks.append(a.energy(assignment,poses)==b.energy(assignment,poses))
    for j in range(a.n):
        for i in range(a.m):checks.append(a.pair_cost(j,i,poses[i])==b.pair_cost(j,i,poses[i]))
poses=[pose(np.zeros(3),a.anchor[i]+C[i]['xyz'].mean(0)-H[i]['xyz'].mean(0),a.anchor[i]) for i in range(a.m)]
aa=a.refine(np.array([0,1]),poses,2);bb=b.refine(np.array([0,1]),poses,2)
checks.extend([aa['assignment']==bb['assignment'],aa['energy']==bb['energy'],np.array_equal(aa['poses'],bb['poses'])])
assert all(checks)
out=Path('results/rigid_identity_cache_check_v1');out.mkdir(exist_ok=False)
save(out/'check.json',{'checks_passed':len(checks),'bitwise_equal_cost_energy_refinement':True,'original_point_queries':a.point_queries,'cached_point_queries':b.point_queries,'cache_hits':b.cache_hits,'scope':'Exact engineering equivalence on actual anonymous point subsets, not algorithm benefit or independent scenes'})
print('CACHE_EQUIVALENT',len(checks),a.point_queries,b.point_queries,b.cache_hits)
