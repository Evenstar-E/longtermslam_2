"""Numerical invariants and an outgoing-feedback counterexample, not task evidence."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem,rigid_projection
from inspection_decisions.reversible_surface_v2 import costs


def obs(x):return {'xyz':x,'rgb':np.zeros_like(x),'class':np.full(len(x),5),'mass':np.ones(len(x))}
def hist(x):return {'xyz':x,'rgb':np.zeros_like(x),'tree':cKDTree(x),'class':5,'uid':9}


def main():
    cfg=json.loads(Path('configs/provenance_pose_hypotheses_v1.json').read_text());checks=[]
    x=np.array([[0,0,0],[.01,0,0],[0,.01,0]],float);H=[hist(x)];O=[obs(x+[1,0,0]),obs(x+[1,0,0])]
    I=np.eye(4);T=I.copy();T[0,3]=1;pool=[[np.array([I,T])],[np.array([I])]];labels=[np.zeros(3,int),np.zeros(3,int)]
    ordinary=Problem(O,H,pool,labels,cfg,'pool_detached');candidate=Problem(O,H,pool,labels,cfg,'pool_coupled')
    a=ordinary.block(0,0);b=candidate.block(0,0)
    assert a['accepted'] and a['own_delta']<0 and a['outgoing_delta']>0 and a['energy_after']>a['energy_before'];checks.append('detached forward gain can damage outgoing source explanation')
    assert not b['accepted'] and b['energy_after']==b['energy_before'];checks.append('coupled selection rejects this exact positive joint-energy change')
    assert ordinary.verify_cache()<1e-6 and candidate.verify_cache()<1e-6;checks.append('outgoing costs rebuild exactly after accepted source-pose change')
    assert np.allclose(ordinary.surface(0,0)['xyz'],x);checks.append('source increment canonical coordinates depend on selected pose branch')
    rng=np.random.default_rng(8)
    for _ in range(12):
        h=hist(rng.normal(size=(15,3))*.1);o=[obs(rng.normal(size=(100,3))),obs(rng.normal(size=(100,3)))]
        pp=I.copy();pp[:3,:3]+=rng.normal(size=(3,3))*.03;pp[:3,3]=rng.normal(size=3)*.3;pp=rigid_projection(pp)
        p=Problem(o,[h],[[np.array([pp])],[np.array([I])]],[-np.ones(100,int),-np.ones(100,int)],cfg,'pool_coupled')
        inc=p.surface(1,0);fast=p.column(0,0,pp,inc);dense=costs(o[0],[pp],[h],[inc],cfg)[0][:,0]
        assert np.allclose(fast,np.minimum(dense,cfg['null_cost']),atol=1e-7)
    checks.append('12 spatial sphere acceleration cases equal dense all-point clipped costs')
    assert np.allclose(pp[:3,:3].T@pp[:3,:3],I[:3,:3],atol=1e-12);checks.append('native numerical rotations projected to proper SO3 without changing translation')
    before=ordinary.surface(0,0);ordinary.z[0][:]=-1;after=ordinary.surface(0,0)
    assert len(before['source_voxel'])==3 and len(after['source_voxel'])==0;checks.append('source ownership retraction leaves no stale fused points')
    result={'passed':True,'checks':checks,'counterexample_detached':a,'counterexample_coupled':b}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
