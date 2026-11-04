import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path('third_party/LivingScenes_cpu_port_v1').resolve()))
import torch
from pytorch3d.ops import knn_points,sample_farthest_points
from pytorch3d.ops.points_alignment import iterative_closest_point,SimilarityTransform

class CPUOpsTests(unittest.TestCase):
    def test_exact_l2_knn_with_neighbor_features(self):
        torch.manual_seed(3);a=torch.randn(2,11,9);b=torch.randn(2,17,9)
        expected=((a[:,:,None]-b[:,None])**2).sum(-1).topk(4,largest=False)
        actual=knn_points(a,b,K=4,return_nn=True)
        torch.testing.assert_close(actual.dists,expected.values,atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.equal(actual.idx,expected.indices))
        torch.testing.assert_close(actual.knn,b[torch.arange(2)[:,None,None],actual.idx])
    def test_variable_lengths_zero_padding(self):
        a=torch.tensor([[[0.],[3.],[99.]]]);b=torch.tensor([[[1.],[4.],[100.]]])
        result=knn_points(a,b,lengths1=torch.tensor([2]),lengths2=torch.tensor([2]),K=4,return_nn=True)
        torch.testing.assert_close(result.dists[0],torch.tensor([[1.,16.,0.,0.],[1.,4.,0.,0.],[0.,0.,0.,0.]]))
    def test_reference_fps_extremes_and_padding(self):
        p=torch.tensor([[[0.,0,0],[1,0,0],[3,0,0]]]);q,idx=sample_farthest_points(p,K=5)
        self.assertEqual(idx.tolist(),[[0,2,1,-1,-1]])
        torch.testing.assert_close(q[0,3:],torch.zeros(2,3))
    def test_knn_gradient_matches_selected_pairs(self):
        a=torch.tensor([[[.3,0.],[2.7,0.]]],requires_grad=True);b=torch.tensor([[[0.,0.],[3.,0.]]])
        knn_points(a,b).dists.sum().backward()
        torch.testing.assert_close(a.grad,torch.tensor([[[.6,0.],[-.6,0.]]]),atol=1e-6,rtol=1e-6)
    def test_official_icp_row_transform_convention(self):
        torch.manual_seed(5);a=torch.randn(1,70,3);t=torch.tensor([[.03,-.02,.01]]);b=a+t[:,None]
        result=iterative_closest_point(a,b)
        torch.testing.assert_close(result.RTs.R,torch.eye(3)[None],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(result.RTs.T,t,atol=1e-5,rtol=1e-5)

if __name__=='__main__':unittest.main()
