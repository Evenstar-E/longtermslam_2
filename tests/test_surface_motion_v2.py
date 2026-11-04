import unittest
import numpy as np
from scripts.surface_motion_v2 import sample_faces,diagnostic,context


class SurfaceMotionTest(unittest.TestCase):
    def test_supported_face_samples(self):
        box=np.array([[-1.,-1,4],[1,1,6]])
        p,n=sample_faces(box,[False,False,False,False,True,False],4)
        self.assertEqual(len(p),16);np.testing.assert_allclose(p[:,2],4);np.testing.assert_allclose(n,np.tile([0,0,-1.],(16,1)))

    def test_plane_tangential_nullspace_not_regularizer_information(self):
        h=w=21;yy,xx=np.mgrid[:h,:w];r=np.stack([(xx-10)/20,(yy-10)/20,np.ones_like(xx)],axis=-1).reshape(-1,3)
        pix=np.flatnonzero((np.abs(xx.ravel()-10)<3)&(np.abs(yy.ravel()-10)<3))
        obs=dict(bounds=np.array([[[-5.,-5,4],[5,5,6]]]),face_support=np.array([[False,False,False,False,True,False]]),rays=r,depth=np.full((h,w),4.),query=[dict(all_pixels=pix)])
        cfg=dict(current_max_points=25,surface_samples_per_edge=4,jacobian_step_m=.002,information_noise_scale_m=.02,maximum_direction_scale_m=.1,relative_eigen_threshold=.02,silhouette_weight=.5)
        ctx=context(obs,0,0,cfg);d=diagnostic(ctx,obs,np.eye(4),np.zeros(3),0,np.zeros(len(ctx['samples']),bool),cfg,'one_sided',np.zeros((6,6)))
        self.assertEqual(d['conditional_rank'],1)
        # With no stable pose support even the normal motion can be camera motion.
        self.assertEqual(d['marginal_rank'],0)


if __name__=='__main__':unittest.main()
