"""Analytic limitation of residual-only non-degradation, not full VGGT-CD run.

Implements only the fixed-scale/rotation fine translation self-check described
in Algorithm 1 of https://arxiv.org/html/2605.16859v1 . No model/dataset ranking.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def main():
    n=110;p=np.column_stack([2*np.arange(n),np.zeros(n),np.zeros(n)]).astype(float);q=p.copy();q[:70,0]+=.5
    tree=cKDTree(q);coarse=np.array([.1,0,0]);before,match=tree.query(p+coarse)
    static=before<3*np.median(before);assert static.sum()>=100
    proposed=np.mean(q[match[static]]-p[static],axis=0);after=tree.query(p+proposed)[0]
    accepted=bool(np.median(after)<np.median(before));final=proposed if accepted else coarse
    result=dict(scope='Analytic audit of fine-stage residual-only self-check, NOT end-to-end VGGT-CD reproduction or benchmark',source='https://arxiv.org/html/2605.16859v1',points=n,unchanged=40,moved=70,coarse_translation=coarse.tolist(),accepted_translation=final.tolist(),static_subset_count=int(static.sum()),median_residual_before_m=float(np.median(before)),median_residual_after_m=float(np.median(after)),true_static_frame_translation=[0,0,0],true_translation_error_before_m=float(np.linalg.norm(coarse)),true_translation_error_after_m=float(np.linalg.norm(final)),accepted=accepted,interpretation='Lower nearest-point residual does not universally imply lower true alignment error; additional assumptions or independent evidence are required.')
    assert accepted and result['true_translation_error_after_m']>result['true_translation_error_before_m']
    out=Path('results/shared_pose_change_v3');out.mkdir(exist_ok=True);(out/'residual_selfcheck_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
