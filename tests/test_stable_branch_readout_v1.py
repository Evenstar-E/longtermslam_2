import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.estimation_v1 import world_components
from inspection_decisions.stable_branch_readout_v1 import transformed_groups, partition_changed


class StablePartitionTests(unittest.TestCase):
    def test_rotation_cannot_repartition_acquired_fragments(self):
        def component(points, pixels):
            points = np.asarray(points, float)
            return dict(appearance_group=0, points_camera=points,
                normals_camera=np.tile([0., 0., 1.], (len(points), 1)),
                normal_valid=np.ones(len(points), bool), pixels=np.asarray(pixels))
        observation = dict(nominal_T=np.eye(4), components=[
            component([[-.3, 0., 0.], [.3, 0., 0.], [0., .3, 0.]], [1, 2, 3]),
            component([[.65, 0., 0.]], [4])])
        inverse = np.eye(4); inverse[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        self.assertEqual(len(world_components(observation)), 1)
        self.assertTrue(partition_changed(observation, inverse))
        stable = transformed_groups(observation, inverse)
        self.assertEqual(len(stable), 1)
        np.testing.assert_array_equal(stable[0]['pixels'], [1, 2, 3, 4])
        original = world_components(observation)[0]['points']
        np.testing.assert_allclose(stable[0]['points'], original@inverse[:3, :3].T)


if __name__ == '__main__': unittest.main()
