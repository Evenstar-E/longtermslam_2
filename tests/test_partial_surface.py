import unittest
import numpy as np
from scripts.partial_surface_scene import intersect
from scripts.partial_surface_estimation import signed_box_distance,evidence


class PartialSurfaceTest(unittest.TestCase):
    def test_ray_front_face_and_miss(self):
        d,f=intersect(np.zeros(3),np.array([[0,0,1],[2,0,1.]]),np.array([-1,-1,4]),np.array([1,1,6]))
        self.assertEqual(d[0],4);self.assertEqual(f[0],4);self.assertTrue(np.isinf(d[1]))

    def test_static_box_different_surface_centres(self):
        box=np.array([[-.5,-.5,4],[.5,.5,5.]])
        front=np.array([[x,y,4] for x in [-.4,0,.4] for y in [-.4,0,.4]])
        side=np.array([[.5,y,z] for z in [4.1,4.5,4.9] for y in [-.4,0,.4]])
        self.assertGreater(np.linalg.norm(front.mean(0)-side.mean(0)),.25)
        np.testing.assert_allclose(signed_box_distance(front,box),0)
        np.testing.assert_allclose(signed_box_distance(side,box),0)

    def test_unsupported_surface_unknown_and_foreground_occlusion(self):
        box=np.array([[-1.,-1,4],[1,1,6]])
        obs=dict(bounds=box[None],face_support=np.array([[True,True,True,True,False,True]]),query=[dict(points=np.array([[0,0,4.]]*10))],rays=np.array([[0.,0,1]]*10),depth=np.array([[2.]*10]))
        cfg=dict(minimum_supported_fraction=.3,change_threshold_m=.12)
        _,known,_,_=evidence(obs,np.eye(4),[0],cfg);self.assertFalse(known[0])
        obs['face_support'][:]=True
        score,known,_,occluded=evidence(obs,np.eye(4),[0],cfg)
        self.assertFalse(known[0]);self.assertEqual(score[0],0);self.assertEqual(occluded,[10])


if __name__=='__main__':unittest.main()
