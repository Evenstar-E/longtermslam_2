import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes, mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.event_likelihood_v7 import held_clouds
from comparative_evidence.ray_likelihood_v8 import forecast, encode_depth, log_likelihood, fantasy_weights, acquisition_scores, render_depth, SYMBOLS


class RayV8Tests(unittest.TestCase):
    def test_actual_observable_edges(self):
        obs = dict(depth=np.array([[0, .01, .5, 31.99, 32., 40.]]))
        np.testing.assert_array_equal(encode_depth(obs, np.arange(6)), [0,1,2,64,65,65])

    def test_unknown_is_uniform_not_absence_and_identical_no_information(self):
        m = dict(prob=np.full((32,5,24,SYMBOLS), 1/SYMBOLS))
        for shared in (False, True):
            np.testing.assert_allclose(log_likelihood(m, np.zeros(24,int),shared), -24*np.log(SYMBOLS))
            np.testing.assert_allclose(fantasy_weights(m,shared), 1/32)
            s=acquisition_scores(m,[np.vstack([np.arange(32)/32,np.full(32,.35)])],np.ones((32,32)),shared)
            self.assertAlmostEqual(s['candidate'],0,places=10);self.assertAlmostEqual(s['risk'],0,places=10)

    def test_known_occluder_changes_depth_not_asset_event(self):
        T=np.eye(4); sensor=dict(image_width=9,image_height=9,focal_px=10)
        d=render_depth(np.array([[0,0,3.],[0,0,2.]]),np.array([[0,0,1.],[0,0,0.]]),np.array([0,-1]),T,sensor)
        self.assertEqual(d[40],2.)
        self.assertTrue(np.isinf(d[0]))

    def test_joint_shared_latent_changes_dependence_not_marginal(self):
        p=np.full((1,2,2,2),.01);p[0,0,:,0]=.99;p[0,1,:,1]=.99
        m=dict(prob=p)
        self.assertGreater(log_likelihood(m,[0,0],True)[0],log_likelihood(m,[0,0],False)[0])
        self.assertLess(log_likelihood(m,[0,1],True)[0],log_likelihood(m,[0,1],False)[0])

    def test_forecast_truth_free_normalized_opposite_view(self):
        ep=Episode(Path('results/comparative_interface_v1/scenes/0000'),4,2)
        h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
        fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
        particles=mixture_particles(fit)[0];records=held_clouds(fit,h,q,meta)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('forecast filesystem read')),patch.object(Path,'read_text',side_effect=AssertionError('forecast filesystem read')):
            m=forecast(('history',6),fit,meta,records,particles)
            empty=forecast(('history',6),fit,meta,[],particles)
        self.assertTrue(np.isfinite(m['prob']).all());np.testing.assert_allclose(m['prob'].sum(-1),1)
        np.testing.assert_allclose(empty['prob'],1/SYMBOLS)
        self.assertTrue(set(m['audit']['source_tokens'])<={'history_00','current_00'})
        obs=ep.act('retrieve_history',6);z=encode_depth(obs,m['pixels'])
        self.assertEqual(z.shape,(len(m['pixels']),));self.assertTrue(np.isfinite(log_likelihood(m,z)).all())


if __name__=='__main__':unittest.main()
