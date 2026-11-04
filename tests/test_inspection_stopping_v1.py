import unittest
from pathlib import Path
from inspection_decisions.costs import prefix_cost,threshold_stop,report_risk
from comparative_evidence.interface import Episode


class StoppingTests(unittest.TestCase):
    def test_stopper_does_not_consume_future(self):
        def stream():
            yield dict(assets=[dict(label=-1,decision_risk=.001)])
            yield dict(assets=[dict(label=0,decision_risk=.01)])
            raise AssertionError('Future prefix read after stop')
        self.assertEqual(threshold_stop(stream(),.02),1)
        self.assertEqual(report_risk(dict(assets=[dict(label=-1,decision_risk=.001)])),.35)

    def test_return_reserve_matches_real_episode(self):
        root=Path('results/comparative_interface_v1/scenes/0000')
        for actions in ([],[('current',2)],[('current',2),('history',3),('current',1)]):
            ep=Episode(root,4,2);ep.act('retrieve_history',0);meta=ep.metadata()
            for epoch,view in actions:ep.act('retrieve_history' if epoch=='history' else 'observe_current',view)
            state=ep.state();c=prefix_cost(state['events'],meta,len(actions),index_bytes=state['index_bytes']);actual=ep.finish()
            self.assertAlmostEqual(c['total_travel_m'],actual['travel_m']);self.assertEqual(c['payload_bytes'],actual['payload_bytes'])
            self.assertEqual(c['history_views'],actual['history_views']);self.assertEqual(c['additional_current_views'],actual['additional_current_views'])
            self.assertIsNone(c['inference_s'])

    def test_invalid_cost_parameters(self):
        for stage,speed,bandwidth in ((-1,1,1),(0,0,1),(0,1,0)):
            with self.assertRaises(ValueError):prefix_cost([],{},stage,speed,bandwidth)


if __name__=='__main__':unittest.main()
