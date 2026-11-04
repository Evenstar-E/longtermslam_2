import unittest
import numpy as np
from scripts.history_modes_v6 import clusters,memory
from scripts.run_history_modes import corrupt

class ModesTest(unittest.TestCase):
    def test_clusters_preserve_two_separated_locations_and_anchor(self):
        p=np.array([[0.,0,0],[.01,0,0],[3,0,0],[3.01,0,0],[3.02,0,0],[3.03,0,0]])
        groups=clusters(p,.36);self.assertEqual(sorted(map(len,groups)),[2,4]);obs=dict(history_positions=p[:,None],history_descriptors=np.zeros((6,1,8)))
        selected,_=memory(obs,'anchor_cluster',dict(cluster_radius_m=.36));self.assertLess(selected[0,0],.02)
        median,_=memory(obs,'geomedian',{});self.assertGreater(median[0,0],2.9)
    def test_corruption_keeps_anchor_and_per_frame_point_set(self):
        p=np.arange(6*8*3).reshape(6,8,3).astype(float);f=np.zeros((6,8,8));obs=dict(history_positions=p,history_descriptors=f)
        out,provenance,_=corrupt(obs,dict(pair_fraction=.5,swapped_views=4,translation_sigma_m=0,rotation_sigma_deg=0),9)
        np.testing.assert_array_equal(out['history_positions'][0],p[0]);np.testing.assert_array_equal(provenance[0],np.arange(8))
        for h in range(6):np.testing.assert_array_equal(out['history_positions'][h],p[h,provenance[h]])
    def test_historical_alignment_repairs_permuted_rigid_frame(self):
        import json
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from scripts.history_alignment_control import infer
        from scripts.joint_estimation import transform
        cfg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text())['algorithm'];sampler=json.loads(Path('configs/joint_proposals_v4.json').read_text());p=np.random.default_rng(5).normal(size=(8,3));f=np.eye(8);E=np.eye(4);E[:3,:3]=Rotation.from_rotvec([.1,.1,-.2]).as_matrix();E[:3,3]=[.2,-.3,.1];order=np.array([3,0,7,4,1,6,5,2])
        obs=dict(history_positions=np.array([p,transform(p[order],E)]),history_descriptors=np.array([f,f[order]]),current_positions=p,current_descriptors=f,initial_T=np.eye(4));pred=infer(obs,cfg,sampler,8)
        np.testing.assert_allclose(pred['T'],np.eye(4),atol=1e-8);np.testing.assert_array_equal(pred['assignment'],np.arange(8));self.assertFalse(pred['changed'].any())

if __name__=='__main__':unittest.main()
