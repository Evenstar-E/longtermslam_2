import unittest
from unittest.mock import patch
import numpy as np
import scripts.map_evidence_scene_v4b as scene


class ConservativeEvidenceTest(unittest.TestCase):
    def test_silhouette_not_promoted_to_world_axis_plane(self):
        endpoints=[dict(source='visible_boundary',tight=True,axis=0,value=0.)]
        with patch.object(scene,'_raw_summaries',return_value=(endpoints,np.zeros((20,3)))):
            result,_=scene.summaries(None,None,None,None,None,None)
        self.assertFalse(result[0]['tight']);self.assertEqual(result[0]['source'],'censored_silhouette')

    def test_weak_plane_support_remains_one_sided(self):
        endpoints=[dict(source='observed_plane',tight=True,axis=0,value=0.)]
        points=np.column_stack([np.linspace(-1,1,20),np.zeros((20,2))])
        with patch.object(scene,'_raw_summaries',return_value=(endpoints,points)):
            result,_=scene.summaries(None,None,None,None,None,None)
        self.assertFalse(result[0]['tight'])


if __name__=='__main__':unittest.main()
