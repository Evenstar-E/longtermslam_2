"""Independent geometry recovery and rejection checks for existing-method implementations."""
import json,unittest
from pathlib import Path
import numpy as np
from scripts.real_batch_baselines import pnp,rigid_ransac
from visual_reinspection.renderer import project,transform
from visual_reinspection.baseline import pose_errors

class BatchGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path("configs/real_batch_v1.json").read_text())["methods"]
        self.rng=np.random.default_rng(42)
        self.a=self.rng.uniform([-1,-.7,2],[1,.7,5],size=(60,3))
        angle=.15;self.T=np.eye(4)
        self.T[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        self.T[:3,3]=[.2,-.05,.1]
    def test_rigid_recovers_pose_with_outliers(self):
        b=transform(self.a,self.T)+self.rng.normal(0,.002,self.a.shape)
        b[:15]=self.rng.uniform([-2,-2,1],[2,2,5],size=(15,3))
        T,meta=rigid_ransac(self.a,b,self.cfg["rgbd"],5)
        te,re=pose_errors(T,self.T)
        self.assertTrue(meta["accepted"]);self.assertLess(te,.02);self.assertLess(re,.5)
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1,places=8)
    def test_pnp_recovers_pose_with_outliers(self):
        K=np.array([[610,0,424],[0,610,240],[0,0,1]],float)
        uv=project(transform(self.a,self.T),K)+self.rng.normal(0,.1,(60,2))
        uv[:12]=self.rng.uniform([0,0],[848,480],size=(12,2))
        T,meta=pnp(self.a,uv,K,self.cfg["pnp"],5)
        te,re=pose_errors(T,self.T)
        self.assertTrue(meta["accepted"]);self.assertLess(te,.02);self.assertLess(re,.5)
        self.assertGreaterEqual(meta["support"],40)
    def test_collinear_correspondences_rejected(self):
        a=np.column_stack([np.arange(8),np.zeros(8),np.ones(8)])
        T,meta=rigid_ransac(a,a,self.cfg["rgbd"],5)
        self.assertIsNone(T);self.assertFalse(meta["accepted"])
    def test_insufficient_matches_rejected(self):
        T,meta=rigid_ransac(self.a[:2],self.a[:2],self.cfg["rgbd"],5)
        self.assertIsNone(T);self.assertFalse(meta["accepted"])
    def test_inconsistent_triangles_rejected_without_nan(self):
        a=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
        b=np.array([[0.,0.,2.],[100.,0.,2.],[0.,100.,2.]])
        T,meta=rigid_ransac(a,b,self.cfg["rgbd"],5)
        self.assertIsNone(T);self.assertFalse(meta["accepted"])
        self.assertEqual(meta["reason"],"no_refittable_consensus")

if __name__=="__main__":unittest.main()

