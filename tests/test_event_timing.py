import json,unittest
from pathlib import Path
from scripts.analyze_event_evidence import event_checks
from scripts.build_event_robust_pairs import select

class EventTimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/event_evidence_v1.json').read_text())
        self.event={'instance':'O(1)','kind':'disappearance','csv_time_ns':10_000_000_000,
          'candidate_times_ns':{'unshifted':10_000_000_000,'plus_bag_start':14_000_000_000},'possible_record_end_censor':False}
    def row(self,t,state):return {'stamp_ns':round(t*1e9),'primary':{'state':state}}
    def test_occluded_after_is_not_observed_disappearance(self):
        rows=[self.row(9.5,'supported_surface'),self.row(10.5,'occluded'),self.row(14.5,'insufficient_projection')]
        out=event_checks(self.event,rows,self.cfg)
        self.assertFalse(out['clocks']['unshifted']['bracketed_diagnostic'])
        self.assertIsNone(out['disagreement_interval']['diagnostically_favors'])
    def test_conflicting_visible_states_do_not_verify_clock(self):
        rows=[self.row(11,'supported_surface'),self.row(11.5,'supported_surface'),self.row(12,'free_space_evidence')]
        self.assertIsNone(event_checks(self.event,rows,self.cfg)['disagreement_interval']['diagnostically_favors'])
    def test_robust_pair_excludes_ambiguous_times_and_keeps_missing(self):
        rows=[self.row(9,'supported_surface'),self.row(12,'supported_surface'),self.row(13,'free_space_evidence'),self.row(15,'free_space_evidence')]
        pre,post=select(self.event,rows)
        self.assertEqual(pre['stamp_ns'],9_000_000_000);self.assertEqual(post['stamp_ns'],15_000_000_000)
        pre,post=select(self.event,rows[1:])
        self.assertIsNone(pre);self.assertIsNotNone(post)

if __name__=='__main__':unittest.main()
