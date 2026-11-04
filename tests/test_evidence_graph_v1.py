import copy,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_terminal
from inspection_decisions.evidence_graph import build_graph,without_nuisance_edges
from inspection_decisions.graph_dataset import collect_scene,read_graph


def fixture(sid,extra=False):
    ep=Episode(Path('results/comparative_interface_v1/scenes')/f'{sid:04d}',4,2)
    h=[ep.act('retrieve_history',0)];q=[ep.preview()]
    if extra:h.append(ep.act('retrieve_history',1))
    meta=ep.metadata();fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
    return ep,h,q,meta,fit,common_terminal(fit)


class EvidenceGraphTests(unittest.TestCase):
    def test_no_file_read_and_unread_index_images_ignored(self):
        ep,h,q,meta,fit,p=fixture(0)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('file read')),patch.object(Path,'read_text',side_effect=AssertionError('file read')):
            g=build_graph(fit,h,q,meta,ep.state(),p)
            other=copy.deepcopy(meta);other['hidden_truth']={'delta':[999]}
            for row in other['history']:
                row['thumbnail_png_base64']='POISON';row['global_rgb_mean']=[999]*3
            self.assertEqual(g,build_graph(fit,h,q,other,ep.state(),p))
        self.assertEqual(len(g['actions']),23)
        self.assertEqual(g['audit']['source_tokens'],['current_00','history_00'])
        json.dumps(g,allow_nan=False)

    def test_shared_frame_blocks_and_ablation(self):
        for sid,coherent in ((0,False),(4,True)):
            ep,h,q,meta,fit,p=fixture(sid,True);g=build_graph(fit,h,q,meta,ep.state(),p)
            pp={r['node']:r for r in g['audit']['patch_sources']}
            edges=[e for e in g['edges'] if e['kind']=='nuisance_coupling' and pp[e['source']]['epoch']=='history' and pp[e['target']]['epoch']=='history' and pp[e['source']]['token']!=pp[e['target']]['token']]
            self.assertTrue(edges)
            self.assertEqual(any(e['features'][-1]>0 for e in edges),coherent)
            self.assertEqual(any(np.linalg.norm(e['features'][:9])>1e-12 for e in edges),coherent)
            ab=without_nuisance_edges(g);self.assertEqual(g['nodes'],ab['nodes']);self.assertEqual(g['actions'],ab['actions'])
            self.assertFalse(any(e['kind']=='nuisance_coupling' for e in ab['edges']))

    def test_future_frame_prior_not_fake_posterior(self):
        ep,h,q,meta,fit,p=fixture(0);g=build_graph(fit,h,q,meta,ep.state(),p)
        nodes={n['id']:n for n in g['nodes']}
        unseen=[n for n in g['audit']['nuisance_keys'] if not n['observed']]
        self.assertEqual(len(unseen),22)
        for row in unseen:
            f=nodes[row['node']]['features'];np.testing.assert_allclose(f[2:9],0);np.testing.assert_allclose(f[9:],1/3)

    def test_collector_separates_labels_and_persists_input_first(self):
        cache=Path('.cache');cache.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='graph_contract_',dir=cache))
        for d in ('inputs/states','estimator_outputs'):(out/d).mkdir(parents=True)
        original_text,original_bytes=Path.read_text,Path.read_bytes
        def text(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth outside label evaluator')
            return original_text(p,*a,**k)
        def data(p,*a,**k):
            if 'evaluation' in p.parts:raise AssertionError('Truth outside label evaluator')
            return original_bytes(p,*a,**k)
        calls=[]
        def label(scene,prediction):
            files=list((out/'inputs'/'states').glob('*.gz'))
            self.assertEqual(len(files),len(calls)+1)
            for file in files:
                g=read_graph(file);self.assertNotIn('metric',g);self.assertNotIn('realized_loss_drop',g)
            calls.append(1)
            return dict(task_loss=0.)
        with patch.object(Path,'read_text',text),patch.object(Path,'read_bytes',data),patch('inspection_decisions.graph_dataset.evaluate',side_effect=label):
            r=collect_scene(Path('results/comparative_interface_v1/scenes/0000'),out,0,'pipeline_test','testfamily',[[['history',1],['current',1]]],[4,2])
        self.assertEqual(len(r['states']),24);self.assertEqual(len(calls),24)
        self.assertEqual(r['solve_count'],24);self.assertEqual(len(r['pair_labels']),1)
        self.assertTrue(all('metric' not in s for s in r['states']))


if __name__=='__main__':unittest.main()
