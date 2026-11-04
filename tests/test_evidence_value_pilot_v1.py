import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from inspection_decisions.graph_dataset import read_graph
from inspection_decisions.value_pilot_v1 import encode_options,fit_value_model,predict_value,choose


class ValuePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph=read_graph('results/evidence_graph_dataset_v1/shard_0/inputs/states/0000_initial.json.gz')
        cls.pairs=json.loads(Path('configs/evidence_graph_dataset_v1.json').read_text())['pairs']

    def test_pure_encoding_ignores_audit_and_order(self):
        clean=encode_options(self.graph,self.pairs)
        graph=copy.deepcopy(self.graph);graph['audit']={'truth':999999,'seed':-12,'split':'evil'}
        graph['nodes'].reverse();graph['edges'].reverse()
        # Anonymous node/edge storage order and metadata are not learned features.
        with patch('builtins.open',side_effect=AssertionError('actor filesystem read')):
            other=encode_options(graph,self.pairs)
        np.testing.assert_allclose(clean['X'],other['X'],atol=1e-10,rtol=1e-10)
        self.assertEqual(clean['X'].shape[0],31)
        self.assertTrue(np.isfinite(clean['X']).all())

    def test_covariance_ablation_keeps_actions_and_cost_features(self):
        a=encode_options(self.graph,self.pairs);b=encode_options(self.graph,self.pairs,False)
        self.assertEqual(a['options'],b['options'])
        np.testing.assert_array_equal(a['coverage'],b['coverage'])
        self.assertTrue(np.any(a['X']!=b['X']))
        np.testing.assert_array_equal(a['X'][:,-1],b['X'][:,-1])

    def test_decomposition_sign_and_stop(self):
        rng=np.random.default_rng(7);X=rng.normal(size=(5,4,6))
        # Gains1,2,pair4 => interaction +1, not -1.
        y=np.tile([0.,1.,2.,4.],(5,1))
        m=fit_value_model('mean_interaction_rbf',X,y,[1,2],[3],[(1,2)],.1,1.)
        self.assertAlmostEqual(m['constant_interaction'],1.)
        p=predict_value(m,X)
        np.testing.assert_allclose(p[:,3]-p[:,1]-p[:,2],1.)
        np.testing.assert_array_equal(p[:,0],0.)
        np.testing.assert_array_equal(choose(np.array([[0.,-.1,-.2],[0.,.02,.03]]),[0,1,2],.05),[0,0])

    def test_training_and_prediction_deterministic_and_nonmutating(self):
        rng=np.random.default_rng(19);X=rng.normal(size=(4,4,7));y=rng.normal(size=(4,4));y[:,0]=0
        before=X.copy();models=[fit_value_model('structured_rbf',X,y,[1,2],[3],[(1,2)],1.,.25) for _ in range(2)]
        np.testing.assert_array_equal(predict_value(models[0],X),predict_value(models[1],X))
        np.testing.assert_array_equal(X,before)


if __name__=='__main__':unittest.main()
