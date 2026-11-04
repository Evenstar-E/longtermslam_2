"""Functional geometry and image-evidence checks for the procedural development sample."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_reinspection.renderer import (Solid, intersect, intrinsics, look_at, render,
                                         historical_roi, unproject, project, transform, path_clear)
from visual_reinspection.baseline import estimate_pose, assess
from visual_reinspection.run import run


class VisualSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.cfg=json.loads(Path("configs/visual_sample_v1.json").read_text(encoding="utf-8"))
        cls.K=intrinsics(cls.cfg)
        cls.Tr=look_at(cls.cfg["reference"]["eye"],cls.cfg["reference"]["look_at"])
        cls.Tq=look_at(cls.cfg["capture"]["eye"],cls.cfg["capture"]["look_at"])
        cls.roi=historical_roi(cls.cfg,cls.Tr)
        cls.ref=render(cls.cfg,cls.Tr,noise_seed=1)
        cls.clear=render(cls.cfg,cls.Tq,noise_seed=2)
        cls.dim=render(cls.cfg,cls.Tq,brightness=.6,noise_seed=3)
        cls.removed=render(cls.cfg,cls.Tq,removed=True,noise_seed=4)
        cls.blocked=render(cls.cfg,cls.Tq,removed=True,occluder="narrow",noise_seed=5)
        cls.T=np.linalg.inv(cls.Tq)@cls.Tr

    def assess(self,frame,robust=True):
        return assess(self.ref["rgb"],self.ref["depth"],frame["rgb"],frame["depth"],
                      self.K,self.roi,self.T,self.cfg["baseline"],robust)[0]

    def test_projection_roundtrip(self):
        points=np.array([[.2,-.3,2.],[-1,.4,3.]])
        uv=project(points,self.K)
        np.testing.assert_allclose(unproject(uv,points[:,2],self.K),points,atol=1e-12)

    def test_camera_is_proper_rotation(self):
        np.testing.assert_allclose(self.Tr[:3,:3].T@self.Tr[:3,:3],np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(self.Tr[:3,:3]),1.)
        np.testing.assert_allclose(transform(np.array([self.cfg["reference"]["look_at"]]),
                                            np.linalg.inv(self.Tr))[0,:2],0,atol=1e-12)

    def test_box_depth_and_miss(self):
        s=Solid("test","box",np.zeros(3),np.ones(3),np.zeros((2,2,3)))
        t,n=intersect(s,np.array([0,0,3.]),np.array([[0,0,-1.],[3,0,-1.]]))
        self.assertAlmostEqual(t[0],2.)
        self.assertTrue(np.isinf(t[1]))
        np.testing.assert_allclose(n[0],[0,0,1])

    def test_sphere_depth(self):
        s=Solid("test","sphere",np.zeros(3),np.zeros(3),np.zeros((2,2,3)),1.)
        t,_=intersect(s,np.array([0,0,3.]),np.array([[0,0,-1.]]))
        self.assertAlmostEqual(t[0],2.)

    def test_unchanged_reprojection(self):
        result=self.assess(self.clear)
        self.assertEqual(result["decision"],"unchanged")
        self.assertGreater(result["visible_fraction"],.95)

    def test_light_only_false_difference_is_corrected(self):
        raw=self.assess(self.dim,False)
        robust=self.assess(self.dim)
        self.assertEqual(raw["decision"],"changed")
        self.assertEqual(robust["decision"],"unchanged")
        np.testing.assert_allclose(robust["gain"],[.6,.6,.6],atol=.025)

    def test_removal_creates_measured_depth_evidence(self):
        result=self.assess(self.removed)
        self.assertGreater(result["p95_depth_residual_m"],.025)
        # Do not encode success as a test: the initial 7% area gate misses this ROI.
        self.assertGreater(result["changed_fraction"],.03)

    def test_occlusion_is_insufficient_not_removed(self):
        result=self.assess(self.blocked)
        self.assertEqual(result["decision"],"insufficient")
        self.assertGreater(result["occluded_fraction"],.95)

    def test_sift_estimates_pose_from_images(self):
        T,stats,_=estimate_pose(self.ref["rgb"],self.ref["depth"],self.clear["rgb"],
                               self.K,self.roi,self.cfg["baseline"])
        self.assertIsNotNone(T)
        self.assertGreaterEqual(stats["inliers"],15)
        self.assertLess(np.linalg.norm(T[:3,3]-self.T[:3,3]),.02)
        angle=np.degrees(np.arccos(np.clip((np.trace(T[:3,:3]@self.T[:3,:3].T)-1)/2,-1,1)))
        self.assertLess(angle,.5)

    def test_unregistrable_image_does_not_claim_unchanged(self):
        T,_,_=estimate_pose(self.ref["rgb"],self.ref["depth"],np.zeros_like(self.clear["rgb"]),
                            self.K,self.roi,self.cfg["baseline"])
        result,_=assess(self.ref["rgb"],self.ref["depth"],self.clear["rgb"],self.clear["depth"],
                        self.K,self.roi,T,self.cfg["baseline"])
        self.assertIsNone(T)
        self.assertEqual(result["decision"],"insufficient")

    def test_collision_checker_rejects_obstruction(self):
        self.assertFalse(path_clear(self.cfg,[[.23,1.47,1.2],[.23,1.47,.2]],True,"narrow",.1))
        self.assertTrue(path_clear(self.cfg,[[0,1.25,2.65],[1.05,1.25,2.65]],True,"narrow",.15))

    def test_existing_output_not_overwritten(self):
        # A known existing directory avoids sandbox-specific temp cleanup permissions.
        with self.assertRaises(FileExistsError):
            run(Path("configs/visual_sample_v1.json"),Path(__file__).resolve().parent)


if __name__=="__main__":
    unittest.main()

