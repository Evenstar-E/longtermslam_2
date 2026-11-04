"""Audit retained hypotheses, then score both common-pool and refined stages."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
import argparse
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_observation
from inspection_decisions.material_correspondence_inference_v3 import inverse
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,nearest_targets,pair_stats,write_csv
from scripts.score_material_joint_descriptor_comparison_v1 import pool_oracle,aggregate
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/material_correspondence_comparison_v3');args=ap.parse_args()
    root=Path(args.root);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'))
    cfg=proto['base_config'];seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    data=Path(proto['config']['dataset']);assert sha(data/'release_manifest.json')==proto['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    rows=[];pairs_all=[]
    for name in proto['scenes']:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(dest)
        records=json.loads((dest/'frames.json').read_text(encoding='utf-8'));pairs=json.loads((dest/'supervision/pairs_v3.json').read_text(encoding='utf-8'))
        split=json.loads((dest/'public_metadata.json').read_text(encoding='utf-8'))['split'];cache={}
        for r in records:
            obs=load_observation(dest/r['observation'],'flat')
            with np.load(dest/r['labels']) as f:gt={k:f[k] for k in ['instance','material_xyz']}
            cache[r['index']]=dict(obs=obs,truth=gt,trees=nearest_targets(obs))
        for v in proto['config']['variants']:
            with np.load(root/name/f'{v}_union_pools.npz') as f:pool=dict(f)
            # Verify EVERY old method's exact two-period and composed poses,
            # not merely an aggregate count claiming old candidates retained.
            for m in ['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
                with np.load(Path('results/textured_material_comparison_v1')/name/f'{v}_{m}.npz') as f:p=dict(f)
                for i,(j,k) in enumerate(p['target_slots']):
                    for pe,a,b,T in [(0,i,j,p['transforms'][i,0]),(1,i,k,p['transforms'][i,1]),
                                     (2,j,k,p['transforms'][i,1]@inverse(p['transforms'][i,0]))]:
                        values=pool['transforms'][pe,a,b,:pool['counts'][pe,a,b]]
                        assert np.min(np.max(abs(values-T),axis=(1,2)))<1e-7;checks+=1
            oracle=pool_oracle(truth,pool,anchors,cfg)
            for m in proto['methods']:
                for stage in ['pooled','refined']:
                    suffix='_pooled' if stage=='pooled' else ''
                    with np.load(root/name/f'{v}_{m}{suffix}.npz') as f:pred=dict(f)
                    assert np.isfinite(pred['transforms']).all();checks+=1
                    for e in [0,1]:assert np.array_equal(np.sort(pred['target_slots'][:,e]),np.arange(24));checks+=1
                    R=pred['transforms'][:,:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
                    rr=task_rows(name,split,v,m,pred,truth,anchors,cfg)
                    for i,r in enumerate(rr):r.update(stage=stage,oracle_pool_worst_error_m=oracle[i],oracle_pool_qualified=int(oracle[i]<=.05),detail=int(i in truth['detail_assets']))
                    rows+=rr
                    if stage=='refined':
                        paths=json.loads((root/name/f'{v}_{m}_optimization.json').read_text(encoding='utf-8'))['paths']
                        assert all(np.max(np.diff(p))<=1e-10 for p in paths if len(p)>1);checks+=24
                        assert np.max(abs(np.array([p[-1] for p in paths])-pred['objectives']))<1e-9;checks+=1
                        alias='persistent_visible' if m not in ['ordinary','soft3_no_visibility'] else m
                        pp=pair_stats(dest,v,alias,pred,records,pairs,cfg,cache)
                        for p in pp:p['method']=m
                        pairs_all+=pp
        print('V3 SCORED',name,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'correspondence_scores.csv',pairs_all)
    result={stage:aggregate([r for r in rows if r['stage']==stage],pairs_all if stage=='refined' else [],proto['methods'],proto['config']['variants'],proto['paired_controls']) for stage in ['pooled','refined']}
    save(root/'scores.json',result)
    save(root/'scoring_audit.json',dict(checks=checks,all_predictions_sealed_before_truth=True,
        old_pose_retention_checked=True,all_refinement_paths_monotone=True,source_sha=sha(__file__),
        task_rows=len(rows),pair_condition_rows=len(pairs_all),hashes={p:sha(root/p) for p in ['scores.json','task_scores.csv','correspondence_scores.csv']}))
    print('V3 SCORING COMPLETE',checks,flush=True)


if __name__=='__main__':main()
