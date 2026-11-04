import numpy as np
from inspection_decisions.visible_sampling_centroid_v1 import front_pixels,residual_moments


def main():
    rng=np.random.default_rng(20261024);checks=0
    for n in [0,1,17,100]:
        obs=rng.normal(size=(n,3));pred=rng.normal(size=(n,3));cell=rng.integers(0,7,n);centers=rng.normal(size=(7,3))
        m=residual_moments(obs,pred,centers,cell,7,.015**2,np.ones(7)*.001)
        assert m['equivalence_error']<1e-8 and m['visible_mean']<=m['pixel']+1e-8;checks+=1
    K=np.eye(3);x=np.array([[1,1,1],[2,2,2],[-1,0,1],[0,0,-1.]])
    ix,u,v=front_pixels(x,K,(3,3));assert np.array_equal(ix,[0]);checks+=1
    # Half of a fixed material cell visible: corrected predicted mean is exact;
    # the full-cell center retains a deterministic nonzero bias.
    full=np.array([[0.,0.,1.]]);visible=np.array([[.01,0,1],[.02,0,1]])
    m=residual_moments(visible,visible,full,np.zeros(2,int),1,.015**2,np.zeros(1))
    assert m['visible_mean']==0 and m['fixed_mean']>0 and m['scatter']==0;checks+=1
    # A coherent observation offset survives centering; no free bias variable.
    m=residual_moments(visible+[.01,0,0],visible,full,np.zeros(2,int),1,.015**2,np.zeros(1))
    assert m['visible_mean']>0 and m['equivalence_error']<1e-12;checks+=1
    print('PASS',checks,'moment, empty-data, self-occlusion and visible-mean properties')


if __name__=='__main__':main()
