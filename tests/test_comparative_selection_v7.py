import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes,mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.event_likelihood_v7 import (held_clouds,feature_forecast,zbuffer_features,
    log_likelihood,fantasy_weights,acquisition_scores)
from comparative_evidence.selection_v5 import common_losses


class SelectionV7Tests(unittest.TestCase):
    def test_known_occluder_blocks_represented_surface(self):
        T=np.eye(4);sensor=dict(image_width=9,image_height=9,focal_px=10)
        points=np.array([[0,0,3.],[0,0,2.]])
        means,counts=zbuffer_features(points,np.array([[0,0,1.],[0,0,0.]]),np.array([0,-1]),T,sensor,1)
        self.assertEqual(counts[0],0)
        means,counts=zbuffer_features(points[:1],np.array([[0,0,1.]]),np.array([0]),T,sensor,1)
        self.assertEqual(counts[0],9)

    def test_empty_event_likelihood_and_identical_states_give_no_information(self):
        model=dict(means=np.zeros((32,3)),probabilities=np.full((32,1),.6),R=np.eye(3),
            broad_R=np.eye(3)*2,broad_mean=np.zeros(3),match_probability=.9)
        np.testing.assert_allclose(log_likelihood(model,[False],np.zeros(3)),np.log(.4))
        W=fantasy_weights(model);np.testing.assert_allclose(W,np.full((12,32),1/32))
        scores=acquisition_scores(model,[np.vstack([np.arange(32)/32,np.full(32,.35)])],np.ones((32,32)))
        self.assertAlmostEqual(scores['risk'],0);self.assertAlmostEqual(scores['candidate'],0,places=12)

    def test_forecast_uses_only_acquired_data_and_shared_noise(self):
        raw=Path.read_bytes;text=Path.read_text
        def read_bytes(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth')
            return raw(p,*a,**k)
        def read_text(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth')
            return text(p,*a,**k)
        with patch.object(Path,'read_bytes',read_bytes),patch.object(Path,'read_text',read_text):
            ep=Episode(Path('results/comparative_interface_v1/scenes/0003'),1,0)
            h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
            fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
            particles=mixture_particles(fit)[0];records=held_clouds(fit,h,q,meta)
            m=feature_forecast(('current',1),fit,meta,records,particles)
            c=feature_forecast(('current',1),fit,meta,records,particles,False)
        np.testing.assert_array_equal(m['means'],c['means']);np.testing.assert_array_equal(m['R'],c['R'])
        np.testing.assert_allclose(c['probabilities'],np.tile(m['probabilities'].mean(0),(32,1)))
        self.assertGreater(np.linalg.eigvalsh(m['R']).min(),0)
        self.assertGreater(np.max(abs(m['R']-np.diag(np.diag(m['R'])))),0)
        self.assertTrue(set(m['audit']['source_tokens'])<= {'history_00','current_00'})
        scores=acquisition_scores(m,*common_losses(fit,mixture_particles(fit)))
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))


if __name__=='__main__':unittest.main()
