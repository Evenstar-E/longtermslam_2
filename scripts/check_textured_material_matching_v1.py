"""Mechanism tests: rigid motion, censoring, and three-way assignment."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.textured_material_matching_v1 import rigid, apply, ray_state, solve_assignments, persistence_loss


def main():
    cfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text())
    rng=np.random.default_rng(42); checks=0
    for _ in range(30):
        a=rng.normal(size=(20,3)); T=np.eye(4)
        T[:3,:3]=Rotation.random(random_state=rng).as_matrix();T[:3,3]=rng.normal(size=3)
        assert np.max(abs(rigid(a,apply(T,a))-T))<1e-12;checks+=1
    frame=dict(C=np.eye(4),K=np.array([[10.,0,5],[0,10,5],[0,0,1]]),
               depth=np.full((11,11),2.),depth_valid=np.ones((11,11),bool),anonymous_slot=np.zeros((11,11),int))
    p=np.array([[0,0,2.],[0,0,3.],[0,0,1.],[5,0,2.],[0,0,-1.]])
    s,g=ray_state(p,[frame],0,cfg)
    assert np.array_equal(s,[1,0,2,0,0]);checks+=5
    # Censoring nearer depth is not a match and invalid depth is not free space.
    frame['depth_valid'][:]=False
    assert (ray_state(p,[frame],0,cfg)[0]==0).all();checks+=5
    c={(0,1):np.ones((24,24)),(0,2):np.ones((24,24)),(1,2):np.ones((24,24))}
    pp1=rng.permutation(24);pp2=rng.permutation(24)
    c[0,1][np.arange(24),pp1]=0;c[0,2][np.arange(24),pp2]=0;c[1,2][pp1,pp2]=0
    a,b=solve_assignments(c,cfg)
    assert np.array_equal(a,pp1) and np.array_equal(b,pp2);checks+=48
    a,b=solve_assignments(c,cfg,np.zeros((24,24,24)))
    assert np.array_equal(a,pp1) and np.array_equal(b,pp2);checks+=48
    x=rng.normal(size=(8,3));T1=np.eye(4);T1[:3,3]=[.2,0,0];T2=np.eye(4);T2[:3,3]=[0,.3,0]
    mm=dict(ia=np.arange(8),ib=np.arange(8))
    val,n=persistence_loss(dict(xyz=x),dict(xyz=apply(T1,x)),dict(xyz=apply(T2,x)),mm,mm,T1,T2,cfg)
    assert val==0 and n==8;checks+=1
    bad,_=persistence_loss(dict(xyz=x),dict(xyz=apply(T1,x)),dict(xyz=apply(T2,x)+1),mm,mm,T1,T2,cfg)
    assert bad>.99;checks+=1
    # Independent ground-truth construction: exact motions must yield zero fixed
    # material displacement error; a 6cm intervention must fail the 5cm gate.
    from scripts.score_textured_material_comparison_v1 import task_rows
    poses=np.tile(np.eye(4),(3,24,1,1));poses[:,:,:3,3]=rng.normal(size=(3,24,3))
    perms=np.array([np.arange(24),rng.permutation(24),rng.permutation(24)])
    G=np.array([[poses[e,i]@np.linalg.inv(poses[0,i]) for e in [1,2]] for i in range(24)])
    pred=dict(target_slots=perms[1:].T,transforms=G,triplet_counts=np.zeros(24))
    truth=dict(poses=poses,permutations=perms,changed_assets=[0,1])
    anchors=[rng.normal(size=(8,3)) for _ in range(24)]
    rows=task_rows('fake','test','rich','fusion',pred,truth,anchors,cfg)
    assert all(r['qualified'] and r['worst_error_m']<1e-12 for r in rows);checks+=24
    pred['transforms'][0,0,:3,3]+=[.06,0,0]
    perturbed=task_rows('fake','test','rich','fusion',pred,truth,anchors,cfg)
    assert perturbed[0]['qualified']==0;checks+=1
    assert abs(perturbed[0]['error_12_m']-.06)<1e-12;checks+=1
    assert sum(r['qualified'] for r in perturbed)==23;checks+=1
    print('PASS',checks,'mechanism checks')


if __name__=='__main__':main()
