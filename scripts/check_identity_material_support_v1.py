import numpy as np
from scripts.check_provenance_pose_hypotheses_v1 import hist,obs
from inspection_decisions.identity_material_support_v1 import Problem


def main():
    import json
    from pathlib import Path
    cfg=json.loads(Path('configs/provenance_pose_hypotheses_v1.json').read_text());I=np.eye(4)
    H=[hist(np.array([[0.,0.,0.]]))];O=[obs(np.array([[2.,0.,0.]])) for _ in range(2)]
    p=Problem(O,H,[[np.array([I])],[np.array([I])]], [np.array([0]),np.array([0])],cfg,True)
    p.C[0][:]=p.null
    z,M,d=p.propose(0,0,np.array([p.null]));assert z[0]==0 and not M[0] and abs(d)<1e-9
    z,M,d=p.propose(0,0,np.array([.1]));assert z[0]==0 and M[0] and d<0
    p.M[0][:]=False;assert len(p.surface(0,0)['xyz'])==0 and p.z[0][0]==0
    p.M[0][:]=True;assert len(p.surface(0,0)['xyz'])==1
    control=Problem(O,H,[[np.array([I])],[np.array([I])]], [np.array([0]),np.array([0])],cfg,False)
    control.C[0][:]=p.null;z,M,d=control.propose(0,0,np.array([p.null]));assert z[0]==0 and M[0]
    q=Problem(O,H,[[np.array([I])],[np.array([I])]], [np.array([0]),np.array([0])],cfg,True)
    old=q.C[1].copy();back,_=q.back(0,0,I,q.z[0],np.array([False]))
    assert back>.6 and q.trial is None and np.array_equal(old,q.C[1])
    assert len(q.surface(0,0)['xyz'])==1
    print('PASS: identity/null separation, admission withdrawal/restoration, supported proposal, tie control, mask-only outgoing dependency, trial rollback')


if __name__=='__main__':main()
