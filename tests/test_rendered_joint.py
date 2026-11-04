import json,unittest
from pathlib import Path
import numpy as np
from scripts.rendered_joint_scene import render,extract
from scripts.joint_estimation_v3 import solve
from scripts.joint_estimation_v2 import infer_decoupled
from visual_reinspection.renderer import look_at

class RenderedTest(unittest.TestCase):
    def test_ray_depth_and_image_only_measurement(self):
        K=np.array([[120.,0,80],[0,120,60],[0,0,1]])
        cfg=dict(width=160,height=120,depth_sigma_m=0,rgb_sigma=0,panel_side_m=1.)
        T=look_at([0,0,4],[0,0,0]);texture=np.tile([230,60,50],(1,2,4,1))
        frame=render(np.array([[0.,0,0]]),texture,T,K,cfg,np.random.default_rng(1))
        self.assertAlmostEqual(float(frame['depth'][60,80]),4.)
        d=extract(frame['rgb'],frame['depth'],K,dict(mask_saturation=.25,mask_min_value=.15,minimum_component_pixels=18))
        self.assertEqual(len(d),1);np.testing.assert_allclose(d[0]['position'],[0,0,4],atol=.03)
        # Occluding plane is nearer in actual z-buffer, not a random observation deletion.
        occ=render(np.array([[0.,0,0]]),texture,T,K,cfg,np.random.default_rng(1),True)
        self.assertAlmostEqual(float(occ['depth'][60,80]),2.)
        self.assertEqual(int(occ['ids'][60,80]),-2)
    def test_v2_solver_parity_on_saved_observations(self):
        cfg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text())['algorithm']
        data=np.load('results/joint_estimation_v1/scenes/0450.npz');obs={k:data[k] for k in data.files if not k.startswith('truth_')}
        a=infer_decoupled(obs,cfg,41);b=solve(obs,cfg,41,'v2')
        for key in ['T','assignment','changed']:np.testing.assert_array_equal(a[key],b[key])

if __name__=='__main__':unittest.main()
