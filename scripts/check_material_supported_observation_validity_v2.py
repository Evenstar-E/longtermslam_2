import numpy as np
from inspection_decisions.material_supported_observation_validity_v2 import validity


def checks():
    cfg={'width':320,'height':240,'focal_px':250.,'depth_noise_bound_m':.002};checks=0
    xx,yy=np.meshgrid(np.linspace(-.3,.3,8),np.linspace(-.3,.3,8));plane=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]
    n=np.array([0.,0.,-1.]);center=plane.mean(0);assert validity(plane,n,center,np.zeros(3),cfg)['resolved'];checks+=1
    line=np.c_[np.linspace(-.3,.3,12),np.zeros(12),np.ones(12)];assert not validity(line,n,line.mean(0),np.zeros(3),cfg)['rank_resolved'];checks+=1
    # Points in a valid 2D ray plane passing through the camera do not define
    # an observable depth-return surface along those same rays.
    rayplane=np.c_[xx.ravel(),np.zeros(xx.size),1+yy.ravel()]
    test=validity(rayplane,np.array([0.,1.,0.]),rayplane.mean(0),np.zeros(3),cfg)
    assert test['rank_resolved'] and not test['transversal'] and not test['resolved'];checks+=1
    assert validity(plane,n,center,np.zeros(3),cfg)['second_singular_value_m']>validity(plane,n,center,np.zeros(3),cfg)['noise_frobenius_bound_m'];checks+=1
    print({'observation_validity_checks':'PASS','checks':checks},flush=True);return {'status':'PASS','checks':checks}


if __name__=='__main__':checks()
