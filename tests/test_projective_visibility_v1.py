import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from inspection_decisions.projective_visibility_v1 import triangles_from_mask,rasterize,frame_loss,optimize
from scripts.run_evidence_value_pilot_v1 import read


class ProjectiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read('configs/projective_visibility_v1.json');self.sensor=dict(image_width=40,image_height=40,focal_px=40.)
        self.T=np.eye(4);self.mask=np.zeros((40,40),bool);self.mask[10:30,10:30]=True
        self.mesh=triangles_from_mask(np.full((40,40),2.),self.T,self.mask,self.sensor,self.cfg)
    def view(self,mask,depth):return dict(T=self.T,mask=mask,depth=depth,distance=distance_transform_edt(~mask))
    def test_raster_depth_and_finite_support(self):
        z=rasterize(self.mesh,self.T,np.zeros(3),self.sensor)
        np.testing.assert_allclose(z[np.isfinite(z)],2.,atol=1e-10)
        self.assertTrue(np.isfinite(z[20,20]));self.assertFalse(np.isfinite(z[2,2]))
    def test_unseen_hole_is_not_filled(self):
        mask=self.mask.copy();mask[17:23,17:23]=False
        mesh=triangles_from_mask(np.full((40,40),2.),self.T,mask,self.sensor,self.cfg)
        z=rasterize(mesh,self.T,np.zeros(3),self.sensor)
        self.assertFalse(np.isfinite(z[20,20]));self.assertTrue(np.isfinite(z[12,12]))
    def test_foreground_boundary_is_censored(self):
        z=rasterize(self.mesh,self.T,np.zeros(3),self.sensor);mask=np.isfinite(z);visible=mask.copy();visible[:,:20]=False
        depth=np.full((40,40),10.);depth[mask]=2.;depth[mask&~visible]=1.
        v=self.view(visible,depth);c,stats=frame_loss(self.mesh,v,np.zeros(3),self.sensor,self.cfg,True,True)
        u,_=frame_loss(self.mesh,v,np.zeros(3),self.sensor,self.cfg,False,True)
        self.assertGreater(stats['censored_pixels'],0);self.assertAlmostEqual(c,0,places=10);self.assertGreater(u,.1)
    def test_real_motion_toward_camera_not_removed_as_occlusion(self):
        z=rasterize(self.mesh,self.T,[0,0,-.2],self.sensor);mask=np.isfinite(z);depth=np.where(mask,z,10.)
        cost,_=frame_loss(self.mesh,self.view(mask,depth),np.zeros(3),self.sensor,self.cfg,True,True)
        self.assertGreater(cost,.1)
    def test_local_search_recovers_visible_finite_patch_translation(self):
        true=np.array([.15,-.15,.15]);z=rasterize(self.mesh,self.T,true,self.sensor);mask=np.isfinite(z);v=self.view(mask,np.where(mask,z,10.))
        fit=optimize(self.mesh,[v],np.zeros(3),self.sensor,self.cfg)
        self.assertLess(np.linalg.norm(np.array(fit['d'])-true),.08)
        self.assertLess(fit['loss'],fit['initial_loss']);self.assertLessEqual(fit['evaluations'],61)
    def test_empty_reference_remains_failed_not_fabricated(self):
        v=self.view(self.mask,np.full((40,40),2.));fit=optimize(np.empty((0,3,3)),[v],np.zeros(3),self.sensor,self.cfg)
        self.assertEqual(fit['stats'][0]['comparable_pixels'],0);self.assertGreater(fit['loss'],20)


if __name__=='__main__':unittest.main()
