import unittest
import numpy as np
from scripts.fragment_adapter_v3 import merge_excess_fragments


class FragmentAdapterTest(unittest.TestCase):
    def test_same_reference_fragments_merge_without_truth_or_mutation(self):
        p=np.array([[-2.6,0,3],[-1.4,0,3],[2.,0,3],[-2,2,3],[2,2,3]])
        centres=np.array([[-2.,0,3],[2,0,3],[-2,2,3],[2,2,3]])
        features=np.eye(2)[[0,0,1,1]]
        obs=dict(bounds=np.stack([centres-.5,centres+.5],axis=1),reference_features=features,initial_T=np.eye(4),rays=p/3,depth=np.full((1,5),3.),query=[dict(points=p[i:i+1],all_pixels=np.array([i]),feature=np.eye(2)[0 if i<3 else 1]) for i in range(5)])
        out,groups=merge_excess_fragments(obs,72)
        self.assertEqual(groups,[[0,1],[2],[3],[4]])
        self.assertEqual(len(out['query']),4);self.assertEqual(len(obs['query']),5)
        np.testing.assert_array_equal(out['query'][0]['all_pixels'],[0,1])


if __name__=='__main__':unittest.main()
