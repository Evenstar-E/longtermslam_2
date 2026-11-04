import numpy as np
from inspection_decisions.material_cycle_bridge_v1 import partial_map,compose_maps,material_losses


def main():
    cfg=dict(sensor_sigma_m=.015,color_sigma=.1,match_squared_radius=9.)
    x=np.array([[0,0,1],[.1,0,1],[0,.1,1],[0,0,1.1]])
    a=(x,np.eye(4,3),np.zeros(4));T=np.eye(4);T[:3,3]=[.2,.3,.1]
    b=(x+T[:3,3],np.eye(4,3),np.zeros(4));direct=partial_map(a,b,T,cfg)
    assert np.array_equal(direct,np.arange(4))
    assert len(np.unique(direct[direct>=0]))==sum(direct>=0)
    assert np.array_equal(compose_maps(np.array([1,-1,0]),np.array([2,0])),[0,-1,2])
    raw,vis,d=material_losses(direct,direct,b,np.ones(4),cfg);assert raw==vis==-1 and d['agreed']==4
    missing=np.full(4,-1);raw,vis,d=material_losses(missing,direct,b,np.zeros(4),cfg);assert raw==1 and vis==0 and d['absent_unknown']==4
    raw,vis,d=material_losses(missing,direct,b,np.ones(4),cfg);assert raw==vis==1
    raw,vis,d=material_losses(direct,missing,b,np.zeros(4),cfg);assert raw==vis==0
    wrong=np.roll(direct,1);raw,vis,d=material_losses(wrong,direct,b,np.zeros(4),cfg);assert raw==vis==1 and d['matched_mismatch']==4
    # Coordinate-frame changes preserve assignment and material loss.
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    aa=(x@R.T,a[1],a[2]);bb=(b[0]@R.T,b[1],b[2]);TT=T.copy();TT[:3,3]=R@T[:3,3]
    assert np.array_equal(partial_map(aa,bb,TT,cfg),direct)
    print('PASS: 9 meaningful matching/censoring/composition/equivariance properties')


if __name__=='__main__':main()
