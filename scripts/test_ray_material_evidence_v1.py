"""Analytical model properties and geometric state boundary checks."""
import numpy as np
from inspection_decisions.ray_material_evidence_v1 import occupancy_log_bf,ray_states

def main():
    count=0
    for occupied in range(4):
        for free in range(4):
            assert abs(occupancy_log_bf(0,0,occupied,free))<1e-12;count+=1
    assert occupancy_log_bf(1,0,1,0)>0;assert occupancy_log_bf(0,1,1,0)<0;count+=2
    K=np.eye(3);depth=np.array([[1.,0.],[1.,1.]]);mask=np.array([[1,1],[0,1]],bool)
    pts=np.array([[0,0,1.],[0,0,.5],[0,0,1.5],[1,0,1.],[0,1,1.],[-1,0,1.],[0,0,-1.]])
    states,*_=ray_states(pts,depth,mask,K,.015,[.001,5])
    assert np.array_equal(states,[1,2,0,0,2,0,0]);count+=7
    # Integrated multi-observation evidence is NON-additive: two pixels from one
    # observation must not be silently treated as two independent historical views.
    once=float(occupancy_log_bf(1,0,1,0));pooled=float(occupancy_log_bf(1,0,2,0));assert not np.isclose(pooled,2*once);count+=1
    # Exchange historical/current roles leaves the shared-vs-independent BF unchanged.
    for a,b,c,d in [(1,2,3,1),(0,0,1,3),(4,0,1,0)]:
        assert abs(occupancy_log_bf(a,b,c,d)-occupancy_log_bf(c,d,a,b))<1e-12;count+=1
    print('PASS',count,'model properties (not empirical validation)')

if __name__=='__main__':main()
