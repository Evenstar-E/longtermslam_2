"""Independent algebra and production-score checks before full inference."""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score,common_search,legal_neighbors
from inspection_decisions.material_shared_correspondence_factor_v5 import material_energies,MaterialView
from inspection_decisions.material_correspondence_inference_v3 import PairScore


def main():
    rng=np.random.default_rng(261030);a=np.exp(rng.normal(size=(3,5,8)));b=np.exp(rng.normal(size=(4,5,8)));w=np.ones(8)/8;alpha=np.ones(5)/5
    point,ind,shared=material_energies(a,b,w,alpha);assert np.min(point-ind)>-1e-12
    for i in range(3):
        for j in range(4):
            exact=-sum(alpha[q]*np.log(sum(w[s]*a[i,q,s]*b[j,q,s] for s in range(8))) for q in range(5))
            assert abs(shared[i,j]-exact)<1e-12
    aa=np.repeat(a.mean(2,keepdims=True),8,2);bb=np.repeat(b.mean(2,keepdims=True),8,2)
    _,i,s=material_energies(aa,bb,w,alpha);assert np.max(abs(i-s))<1e-12
    _,i,s=material_energies(a,np.ones_like(b),w,alpha);assert np.max(abs(i-s))<1e-12
    assert np.max(abs(material_energies(np.ones_like(a),np.ones_like(b),w,alpha)[2]))<1e-12
    x=rng.normal(size=(30,3))*.05+[0,0,2];y=x+rng.normal(size=x.shape)*.002
    ga=dict(xyz=x,area=np.ones(30)/30,sigma=np.full(30,.02));gb=dict(xyz=y,area=np.ones(30)/30,sigma=np.full(30,.02))
    A=dict(xyz=x,points=x,tree=cKDTree(x));B=dict(xyz=y,points=y,tree=cKDTree(y));mm=dict(ia=np.arange(30),ib=np.arange(30),distance=np.ones(30)*.2)
    score=PairScore(A,B,mm,ga,gb,{},[],0,{})
    poses=np.tile(np.eye(4),(41,1,1));poses[:,:3,3]=rng.normal(size=(41,3))*.02
    assert np.max(abs(ordinary_score(score,poses)-score.score(poses,['ray'])['ray']))<1e-12
    source=dict(nodes=x[:5,None].repeat(8,1),sigma=np.ones(5)*.02)
    bank=dict(xyz=y,desc=np.ones((30,4))/2,sigma=np.ones(30)*.02,area=np.ones(30)/30)
    view=MaterialView(source,np.ones((5,4))/2,bank,[],0,dict(uniform_floor=.1,sigma_floor_m=.006));assert np.allclose(view.likelihood(poses),1.)
    frame=dict(depth=np.full((64,64),2.),depth_valid=np.ones((64,64),bool),anonymous_slot=np.zeros((64,64),int),C=np.eye(4),K=np.array([[200.,0,32],[0,200.,32],[0,0,1]]))
    score.frames=[frame];assert np.max(abs(ordinary_score(score,poses)-score.score(poses,['ray'])['ray']))<1e-12
    view.frames=[frame];assert np.max(abs(view.likelihood(poses)-1))<1e-12
    # Generic solver automatically finds legal improving swaps, not a supplied
    # GT move, and every objective sees the same state graph.
    n=4;initial=np.c_[np.arange(n),np.roll(np.arange(n),1)];allowed=[[set(range(n)) for _ in range(n)] for _ in range(2)]
    costs=[np.abs(np.arange(n)[:,None]-np.arange(n)[None]) for _ in range(3)]
    def objective(i,j,k):return np.full(4,abs(i-j)+abs(i-k)),None
    result=common_search([initial],objective,allowed,costs,allowed,rounds=4,per_round=100)
    assert np.all(result['costs'][result['best'],np.arange(4)]==0)
    print('PASS 12 independent shared integrals, Jensen, uniform/missing/single-point limits, exact batched ray parity, legal joint swaps',flush=True)


if __name__=='__main__':main()
