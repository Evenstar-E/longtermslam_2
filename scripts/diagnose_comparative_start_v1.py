"""Posthoc same-data full-evidence initialization audit; not policy comparison."""
import hashlib
import json
import time
from pathlib import Path

from comparative_evidence.interface import Episode
from comparative_evidence.estimation_start_audit_v1 import estimate, serializable_prediction
from comparative_evidence.evaluation_v1 import score_prediction


def main():
    out = Path('results/comparative_start_audit_v1')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(); records = []; begin = time.perf_counter()
    for sid in range(48):
        root = Path('results/comparative_interface_v1/scenes')/f'{sid:04d}'
        ep = Episode(root, 12, 11)
        history = [ep.act('retrieve_history', i) for i in range(12)]
        current = [ep.preview()]+[ep.act('observe_current', i) for i in range(1, 12)]
        fitted = estimate(history, current, ep.metadata(), warm_start=True)
        p = serializable_prediction(fitted)
        scores = score_prediction(root, p)
        records.append(dict(scene=sid, **scores, prediction=p, cost=ep.finish()))
        (out/'checkpoint.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    totals = {k:sum(r[k] for r in records) for k in ('changed','tp','qualified_tp','fp','known','wrong_known')}
    # Replay default=False on prespecified scene1 proves isolated copy preserves
    # the original computation when the initializer is disabled.
    root = Path('results/comparative_interface_v1/scenes/0001'); ep = Episode(root, 12, 11)
    h=[ep.act('retrieve_history',i) for i in range(12)]
    q=[ep.preview()]+[ep.act('observe_current',i) for i in range(1,12)]
    default=serializable_prediction(estimate(h,q,ep.metadata(),warm_start=False))
    old=json.loads(Path('results/comparative_selection_v1/predictions/0001_full_evidence.json').read_text())['prediction']
    assert all(default[k]==old[k] for k in default)
    sources=['comparative_evidence/estimation_v1.py','comparative_evidence/estimation_start_audit_v1.py',
             'scripts/diagnose_comparative_start_v1.py']
    result=dict(stage='Posthoc conventional initializer audit; reused full-evidence traces, not candidate confirmation',
        summary=totals, seconds=time.perf_counter()-begin, records=records, default_copy_replay_scene1=True,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=totals,seconds=result['seconds'])))


if __name__=='__main__':main()
