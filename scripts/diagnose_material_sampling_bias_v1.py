"""Posthoc real-data diagnostic for a prospective observation model, not inference.

Reference object frames define fixed 40mm cells. No pose truth enters any selector.
Centroid shifts combine visibility, sampling, sensor and reference-pose errors.
"""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('data/ycbineoat_motion_prepared_v2');out=Path('results/material_cycle_bridge_v1')
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    size=.04;minimum=16;frames={};rows=[]
    for r in read(root/'public.json')['clouds']:
        stem=r['stem'];pose=np.loadtxt(root/'evaluation/poses'/(stem+'.txt')).reshape(4,4)
        with np.load(root/'public/clouds'/(stem+'.npz')) as z:x=z['xyz'].astype(float)
        x=(x-pose[:3,3])@pose[:3,:3];keys=np.floor(x/size).astype(int)
        unique,inv=np.unique(keys,axis=0,return_inverse=True);count=np.bincount(inv);means=np.column_stack([np.bincount(inv,weights=x[:,k])/count for k in range(3)])
        frames[stem]={tuple(k):(int(n),mean) for k,n,mean in zip(unique,count,means)}
    for i,pair in enumerate(read(root/'public.json')['pairs']):
        h=frames[pair['history_stem']];c=frames[pair['current_stem']];cells=[]
        for key in sorted(set(h)&set(c)):
            if min(h[key][0],c[key][0])<minimum:continue
            cells.append(dict(cell=[int(k) for k in key],history_points=h[key][0],current_points=c[key][0],canonical_centroid_shift_mm=float(np.linalg.norm(h[key][1]-c[key][1])*1000)))
        values=[v['canonical_centroid_shift_mm'] for v in cells]
        rows.append(dict(pair=f'pair_{i:02d}',eligible_cells=len(cells),median_mm=float(np.median(values)) if values else None,p95_mm=float(np.quantile(values,.95)) if values else None,over_10mm=sum(v>10 for v in values),cells=cells))
    values=[v['canonical_centroid_shift_mm'] for r in rows for v in r['cells']]
    summary=dict(cell_pair_count=len(values),median_mm=float(np.median(values)),p95_mm=float(np.quantile(values,.95)),over_10mm=sum(v>10 for v in values))
    save(out/'sampling_bias_diagnosis.json',dict(cell_size_m=size,minimum_points_per_cell_per_view=minimum,rows=rows,summary=summary,scope='POSTHOC, canonical cells use evaluator poses. These are view-dependent observed centroid shifts, not independently measured physical displacement or causal isolation of visibility. Correlated cell/pair reuse. Fixed40mm scale, no parameter sweep.'))
    print(summary);print([(r['pair'],r['eligible_cells'],r['median_mm']) for r in rows])


if __name__=='__main__':main()
