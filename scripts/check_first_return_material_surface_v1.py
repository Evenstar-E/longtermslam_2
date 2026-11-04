"""Analytic ray/continuous-triangle and shared-noise geometry checks."""
import numpy as np
from inspection_decisions.first_return_material_surface_v1 import Patch,surface_violation,material_envelope
from scipy.spatial.transform import Rotation


def main():
    cfg={'height':5,'width':5,'focal_px':5.,'depth_tolerance_m':.006,'depth_noise_bound_m':.002}
    patch=Patch([[-2,-2,2],[2,-2,2],[0,2,2]],[[0,1,2]])
    frame={'depth':np.full((5,5),3.),'C':np.eye(4)}
    n,gap=surface_violation(patch,np.eye(4),[frame],cfg);assert n>0 and abs(gap-1)<1e-6
    M=np.eye(4);M[2,3]=2
    assert surface_violation(patch,M,[frame],cfg)[0]==0 # behind may be occluded
    frame['depth'][:]=2.;assert surface_violation(patch,np.eye(4),[frame],cfg)[0]==0
    # Geometry is not snapped to voxel coordinates.
    M[2,3]=.017;frame['depth'][:]=2.017
    assert surface_violation(patch,M,[frame],cfg)[0]==0
    rng=np.random.default_rng(42);checks=4
    for j in range(100):
        A=np.eye(4);B=np.eye(4);A[:3,:3]=Rotation.random(random_state=rng).as_matrix();B[:3,:3]=Rotation.random(random_state=rng).as_matrix()
        A[:3,3]=rng.normal(size=3);B[:3,3]=rng.normal(size=3)
        anchor=rng.normal(size=3);r=rng.normal(size=3)
        env=material_envelope([[{'T':A}],[{'T':B}]],anchor,{'ray':r.tolist()},np.ones((1,1),bool),cfg)
        for eps in [-.002,0.,.002]:
            a=anchor+eps*r;d=(B[:3,:3]-A[:3,:3])@a+B[:3,3]-A[:3,3]
            assert np.all(d>=np.array(env['lower'])-1e-12) and np.all(d<=np.array(env['upper'])+1e-12)
            checks+=1
    print({'checks':checks,'status':'PASS'})


if __name__=='__main__':main()
