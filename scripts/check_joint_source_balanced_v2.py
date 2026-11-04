import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.joint_source_regret_v1 import joint_icp
from inspection_decisions.joint_source_balanced_v2 import joint_icp_balanced


def main():
    rng=np.random.default_rng(20261024);x=rng.normal(size=(100,3))*.2;truth=[np.eye(4)];frames=[SimpleNamespace(xyz=x,rgb=np.zeros_like(x))]
    for j in range(3):
        q=np.eye(4);q[:3,:3]=Rotation.from_rotvec([.05*j,.03,-.02]).as_matrix();q[:3,3]=[.1*j,.05,-.02];truth.append(q);frames.append(SimpleNamespace(xyz=moved(x,np.linalg.inv(q)),rgb=np.zeros_like(x)))
    initial=[q.copy() for q in truth]
    for j in range(1,4):initial[j][:3,3]+=[.003*j,-.002,.001]
    cfg=dict(solver_points_per_view=1500,solver_sweeps=12,correspondence_radius_m=.06,robust_scale_m=.015,color_scale=.1)
    errors={}
    for method,fun in [('union',joint_icp),('balanced',joint_icp_balanced)]:
        last=fun(frames,initial,cfg)[-1]['Q'];assert np.array_equal(last[0],np.eye(4));errors[method]=float(max(np.max(np.abs(np.array(a)-b)) for a,b in zip(last,truth)))
    assert errors['balanced']<1e-4
    print('COUPLED_UNIT',errors,'PASS balanced<1e-4; union retained as weaker ablation')


if __name__=='__main__':main()
