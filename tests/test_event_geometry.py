import unittest
import numpy as np
from scripts.event_geometry import quaternion_rotation,rigid_matrix,world_mesh,projected_surface,depth_evidence

class EventGeometryTests(unittest.TestCase):
    def test_original_optical_axis_faces_body_forward(self):
        R=quaternion_rotation([.5,-.5,.5,-.5])
        np.testing.assert_allclose(R@np.array([0,0,1]),[1,0,0],atol=1e-12)
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
    def test_known_projection_and_z_buffer(self):
        K=np.array([[100,0,50],[0,100,40],[0,0,1.]])
        T=rigid_matrix([1,0,0],[0,0,0,1])
        points=np.array([[1,0,2],[1,0,3],[1.2,0,2],[1,0,-2],[100,0,2]])
        xy,z=projected_surface(points,T,K,100,80)
        self.assertEqual(set(map(tuple,xy)),{(50,40),(60,40)})
        np.testing.assert_allclose(z,2)
    def test_box_frame_rotation_and_translation(self):
        box={'world_R_center':{'x':0,'y':0,'z':np.sqrt(.5),'w':np.sqrt(.5)},'world_P_center':[3,4,5]}
        np.testing.assert_allclose(world_mesh([[1,0,0]],box),[[3,5,5]],atol=1e-12)
    def test_occlusion_is_distinct_from_free_space(self):
        cfg={'minimum_projected_pixels':1,'minimum_support_pixels':1,'invalid_fraction':.5,'support_fraction':.25,'free_fraction':.6,'free_max_support_fraction':.1,'occluded_fraction':.6}
        xy=np.array([[0,0]]);z=np.array([2.]);sem=np.array([[5]])
        self.assertEqual(depth_evidence(xy,z,np.array([[1.]]),sem,5,.05,cfg)['state'],'occluded')
        self.assertEqual(depth_evidence(xy,z,np.array([[3.]]),sem,5,.05,cfg)['state'],'free_space_evidence')
        self.assertEqual(depth_evidence(xy,z,np.array([[2.]]),sem,5,.05,cfg)['state'],'supported_surface')
        self.assertEqual(depth_evidence(xy,z,np.array([[np.nan]]),sem,5,.05,cfg)['state'],'invalid_depth')
    def test_no_projection_is_unresolved(self):
        r=depth_evidence(np.empty((0,2),int),np.array([]),np.zeros((1,1)),np.zeros((1,1)),5,.05,{})
        self.assertEqual(r['state'],'insufficient_projection')

if __name__=='__main__':unittest.main()
