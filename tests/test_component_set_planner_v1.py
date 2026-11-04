import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from inspection_decisions.graph_dataset import read_graph
from inspection_decisions.component_set_v1 import representation,rank_objective,fit,decide,available_pairs


class ComponentSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph=read_graph('results/evidence_graph_dataset_v1/shard_0/inputs/states/0000_initial.json.gz')
        cls.pairs=json.loads(Path('configs/evidence_graph_dataset_v1.json').read_text())['pairs']

    def test_permutation_invariance_and_no_files(self):
        a=representation(self.graph,self.pairs,True);g=copy.deepcopy(self.graph)
        g['audit']={'truth':9999};g['nodes'].reverse();g['edges'].reverse()
        with patch('builtins.open',side_effect=AssertionError()),patch('io.open',side_effect=AssertionError()):b=representation(g,self.pairs,True)
        for key in ('set','additive','pooled','coverage'):np.testing.assert_allclose(a[key],b[key],atol=1e-10,rtol=1e-10)
        self.assertEqual(a['set'].shape,(31,49));np.testing.assert_array_equal(a['set'][0],0)

    def test_pair_budget_and_gradient(self):
        g=copy.deepcopy(self.graph);g['context']['current_remaining']=1
        pairs=available_pairs(g,self.pairs)
        self.assertTrue(all(sum(a[0]=='current' for a in p)<=1 for p in pairs))
        rng=np.random.default_rng(33);D=rng.normal(size=(20,6));w=rng.normal(size=6);sign=rng.choice([-1,1],20);weight=np.ones(20)/20;cost=rng.normal(size=20)
        v,grad=rank_objective(w,D,sign,weight,cost,.1)
        numeric=approx_derivative(lambda z:rank_objective(z,D,sign,weight,cost,.1)[0],w).ravel()
        np.testing.assert_allclose(grad,numeric,atol=1e-7,rtol=1e-7)

    def test_distinguishes_complement_from_repeated_component(self):
        nodes=[dict(id=i,kind='asset',features=[0,0,0,.1,.1,.1,1,.1,-1,.35]) for i in (0,1)]
        nodes += [dict(id=i,kind='patch',features=[1,0,0,0,0,0,1,1,1]) for i in (2,3)]
        nodes += [dict(id=i,kind='action',features=[0]*13) for i in (4,5)]
        edges=[dict(kind='patch_asset',source=2,target=0,features=[1]),dict(kind='patch_asset',source=3,target=1,features=[1])]
        edges += [dict(kind='action_patch',source=a,target=p,features=[float(p==2),1,1,1]) for a in (4,5) for p in (2,3)]
        graph=dict(nodes=nodes,edges=edges,actions=[dict(epoch='history',view=1,node=4),dict(epoch='history',view=3,node=5)],context=dict(history_remaining=3,current_remaining=2,mean_declared_report_risk=.35))
        pairs=[(('history',1),('history',3))];same=representation(graph,pairs)
        for e in graph['edges']:
            if e['kind']=='action_patch' and e['source']==5:e['features'][0]=float(e['target']==3)
        different=representation(graph,pairs)
        np.testing.assert_allclose(same['additive'],different['additive'])
        self.assertGreater(different['set'][-1,0],same['set'][-1,0])
        self.assertEqual(same['coverage'][-1],1);self.assertEqual(different['coverage'][-1],2)

    def test_fit_and_stop_are_deterministic(self):
        groups=[]
        for i in range(3):groups.append(dict(set=np.array([[0.,0.],[1.,0.],[0.,1.]]),gain=np.array([0.,1.,-1.]),counts=np.array([0,1,1])))
        model=fit(groups,'set_rank',.1,.05);again=fit(groups,'set_rank',.1,.05)
        np.testing.assert_array_equal(model['w'],again['w'])
        self.assertEqual(decide(groups[0],model)[0],1)
        self.assertEqual(decide(dict(set=np.zeros((1,2)),counts=np.array([0])),model)[0],0)


if __name__=='__main__':unittest.main()
