from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha

def main():
    root=Path('results/completion_global_bounds_v1');base=Path('results/completion_compatible_anchor_v1')
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    rows=[];truth={}
    for r in read(root/'predictions.json'):
        scene=r['scene']
        if scene not in truth:
            with (base/scene/'evaluation_only.pkl').open('rb') as f:truth[scene]=pickle.load(f)
        d=truth[scene]['anchor_displacement'][r['asset']];lo=np.array(r['lower']);hi=np.array(r['upper'])
        contained=bool(np.all(d>=lo-1e-7) and np.all(d<=hi+1e-7))
        error=float(np.linalg.norm((lo+hi)/2-d))
        rows.append({**r,'contains_truth':contained,'center_error_m':error,'wrong_report':bool(r['report'] and error>.05+1e-7)})
    summary={}
    for method in ['pair','full']:
        rr=[r for r in rows if r['method']==method]
        summary[method]={'n':len(rr),'reports':sum(r['report'] for r in rr),'wrong_reports':sum(r['wrong_report'] for r in rr),
                         'contains_truth':sum(r['contains_truth'] for r in rr),'all_extrema_solved':sum(r['all_extrema_solved'] for r in rr),
                         'median_radius_m':float(np.median([r['radius_m'] for r in rr]))}
    a=[r for r in rows if r['method']=='pair'];b=[r for r in rows if r['method']=='full']
    assert [(r['scene'],r['asset']) for r in a]==[(r['scene'],r['asset']) for r in b]
    summary['full_vs_pair_different_boxes']=sum(not(np.allclose(x['lower'],y['lower'],atol=1e-7) and np.allclose(x['upper'],y['upper'],atol=1e-7)) for x,y in zip(a,b))
    assert all(r['contains_truth'] for r in rows)
    save(root/'scores.json',rows);save(root/'summary.json',summary)
    save(root/'evidence_index.json',{'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evidence_index.json'}})
    print(summary)

if __name__=='__main__':main()
