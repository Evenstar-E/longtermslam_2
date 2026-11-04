import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.interface import components
from comparative_evidence.scene import PALETTE
from inspection_decisions.projective_visibility_v1 import capture
from inspection_decisions.se3_visibility_v1 import proposals,refine_pose,centre_matches,readout,valid_pose
from scripts.joint_estimation import transform
from scripts.run_evidence_value_pilot_v1 import read


class SE3Tests(unittest.TestCase):
    def setUp(self):self.cfg=read('configs/se3_visibility_v1.json')
    def test_full_rigid_registration_reopens_permuted_repeated_identities(self):
        rng=np.random.default_rng(42);X=np.array([[x,y,5+.2*((j+1)%3)] for j,(x,y) in enumerate([(x,y) for x in [-2,-.6,.9,2.1] for y in [-.8,.8]])])
        G=np.eye(4);G[:3,:3]=Rotation.from_euler('xyz',[1.,-1.5,7.],degrees=True).as_matrix();G[:3,3]=[.3,-.2,.1]
        refs=[]
        for x in X:refs.append(dict(group=0,centre=x,points=x+rng.normal(0,.035,(40,3))))
        perm=[5,2,7,0,3,1,6,4];cs=[]
        for j in perm:
            d=np.array([.25,.1,0]) if j==3 else np.zeros(3);p=transform(refs[j]['points']+d,G)
            cs.append(dict(group=0,centre=np.median(p,0),points=p))
        starts,_=proposals(refs,[cs],self.cfg);solutions=[refine_pose(s,refs,[cs],self.cfg) for s in starts];r=min(solutions,key=lambda r:r['score']);H=np.array(r['G'])
        self.assertEqual(dict(r['matches'][0]),dict(enumerate(perm)))
        self.assertLess(np.linalg.norm(H[:3,3]-G[:3,3]),.05)
        self.assertLess(np.rad2deg(Rotation.from_matrix(H[:3,:3]@G[:3,:3].T).magnitude()),1.)
    def test_frame_correction_preserves_raw_pixel_geometry_and_archive_delta(self):
        G=np.eye(4);G[:3,:3]=Rotation.from_euler('xyz',[4,-3,9],degrees=True).as_matrix();G[:3,3]=[.4,-.2,.1]
        x=np.array([[1.,2.,5.]]);delta=np.array([[.1,-.3,.2]])
        np.testing.assert_allclose(transform(transform(x+delta,G),np.linalg.inv(G))-x,delta,atol=1e-12)
        sensor=dict(image_width=32,image_height=32,focal_px=40.,minimum_component_pixels=12,points_per_component=96)
        rgb=np.full((32,32,3),25,np.uint8);rgb[10:22,10:22]=PALETTE[0];depth=np.full((32,32),10.,np.float32);depth[10:22,10:22]=2.
        h=dict(token='history_00',epoch='history',view_id=0,nuisance_group='h',rgb=rgb,depth=depth,nominal_T=np.eye(4),components=components(rgb,depth,sensor))
        q=dict(token='current_00',epoch='current',view_id=0,nuisance_group='c',rgb=rgb,depth=depth,nominal_T=G,components=components(rgb,depth,sensor))
        raw=capture([h],[q],sensor);r=readout(raw,G,[[(0,0)]],self.cfg)
        self.assertEqual(r['prediction']['delta_frame'],'archive');self.assertEqual(r['prediction']['assets'][0]['label'],0)
        np.testing.assert_allclose(r['map_transform_current_from_archive'],G,atol=1e-8)
        self.assertEqual(raw['current'][0]['nominal_T'],G.tolist())
    def test_public_se3_prior_rejects_outside_rotation(self):
        G=np.eye(4);G[:3,:3]=Rotation.from_euler('z',30,degrees=True).as_matrix()
        self.assertFalse(valid_pose(G,self.cfg));self.assertTrue(valid_pose(np.eye(4),self.cfg))
    def test_geometry_reassociation_is_one_to_one(self):
        refs=[dict(group=0,centre=np.array([x,0,3.])) for x in [0.,1.,2.]]
        frames=[[dict(group=0,centre=np.array([x,0,3.])) for x in [2.,0.,1.]]]
        matches,_=centre_matches(np.eye(4),refs,frames,self.cfg)
        self.assertEqual(matches[0],[(0,2),(1,0),(2,1)])


if __name__=='__main__':unittest.main()
