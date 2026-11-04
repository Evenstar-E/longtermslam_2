"""Replay ordinary legal swaps on a previously observed failure; no GT read."""
import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_joint_id_pose_search_v5 import legal_neighbors
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    root=Path('results/material_joint_shared_comparison_v5');parent=Path('results/material_continuous_correspondence_comparison_v4/test_000');variant='changed'
    assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
    with np.load(parent/f'{variant}_surface.npz') as f:initial=f['target_slots']
    with np.load(root/'test_000'/f'{variant}_joint_cache.npz') as f:cache=dict(f)
    lookup={tuple(k):v for k,v in zip(cache['triples'],cache['costs'][:,0])};seeds=[]
    for m in ['baseline','ncc','random','dense','surface','geo','union']:
        with np.load(parent/f'{variant}_{m}.npz') as f:seeds.append(f['target_slots'])
    allowed=[[{int(s[i,e]) for s in seeds} for i in range(24)] for e in [0,1]]
    proxy=[np.zeros((24,24)) for _ in range(3)]
    def value(s):
        keys=[(i,int(j),int(k)) for i,(j,k) in enumerate(s)]
        return sum(lookup[k] for k in keys) if all(k in lookup for k in keys) else None
    before=value(initial);records=[];missing=0;known=None
    for _,__,s in legal_neighbors(initial,allowed,proxy,allowed):
        after=value(s)
        if after is None:missing+=1;continue
        changed=np.flatnonzero((s!=initial).any(1)).tolist();record=dict(affected=changed,before=before,after=after,delta=after-before)
        records.append(record)
        # Expected output from the archived failure, not used to propose/rank.
        if changed==[8,11] and s[11,1]==7 and np.array_equal(s[:,0],initial[:,0]):known=record
    assert known is not None and known['delta']<-.018,known
    best=min(records,key=lambda r:r['after']);assert best['after']<before
    save(root/'real_search_regression.json',dict(complete=True,source_sha=sha(__file__),no_GT_read=True,
        postseal_regression_not_new_task_predictions=True,scope='all cached legal old-link swaps from archived surface initialization; uncached states excluded and counted',
        cached_swaps=len(records),uncached_swaps=missing,best=best,archived_negative_swap=known))
    print('PASS REAL ORDINARY SEARCH REGRESSION',len(records),'cached swaps; best',best,'archived',known,flush=True)


if __name__=='__main__':main()
