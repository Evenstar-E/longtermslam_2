import io,json,unittest
import ijson
from scripts.temporal_metrics import deduplicate,derive,prf

class TemporalMetricTests(unittest.TestCase):
    def test_unassociated_misses_not_erased(self):
        row={k:'0' for k in ['AppearedTP','DisappearedTP','AppearedFP','DisappearedFP','AppearedFN','DisappearedFN','AppearedHallucinatedP','DisappearedHallucinatedP','AppearedMissedP','DisappearedMissedP','NumObjDetected','NumObjHallucinated','NumObjMissed']}
        row.update(Name='3',Query='91054109999',DisappearedTP='3',AppearedFP='8',DisappearedFP='3',DisappearedFN='1',DisappearedHallucinatedP='5',DisappearedMissedP='11')
        metrics=derive(row)
        self.assertAlmostEqual(metrics['change_conditional']['recall'],.75)
        self.assertAlmostEqual(metrics['change_inclusive']['recall'],.2)
        self.assertAlmostEqual(metrics['change_inclusive']['precision'],3/19)
    def test_duplicate_query_not_new_trial(self):
        a={'Name':'1','Query':'20','TP':'4'};b={'Name':'1','Query':'10','TP':'3'}
        rows,count=deduplicate([a,a,b])
        self.assertEqual(count,1);self.assertEqual(len(rows),2)
    def test_conflicting_duplicate_refused(self):
        with self.assertRaises(ValueError):deduplicate([{'Name':'0','Query':'1','n':'3'},{'Name':'0','Query':'1','n':'4'}])
    def test_zero_denominator_is_not_perfect(self):
        m=prf(0,0,3)
        self.assertIsNone(m['precision']);self.assertEqual(m['recall'],0)
        self.assertEqual(m['count_f1'],0);self.assertIsNone(prf(0,0,0)['recall'])
    def test_presence_sentinel_exact_uint64(self):
        raw=b'{"nodes":[{"last_observed_ns":[18446744073709551615]}]}'
        node=next(ijson.items(io.BytesIO(raw),'nodes.item',use_float=False))
        self.assertEqual(node['last_observed_ns'][0],2**64-1)

if __name__=='__main__':unittest.main()
