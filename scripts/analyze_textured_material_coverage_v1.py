"""Post-seal held-out support analysis; labels never return to inference."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import csv
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_matching_v1 import extract_epoch,apply,ray_state
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    root=Path('results/textured_material_comparison_v1')
    assert (root/'prediction_seal.json').exists() and (root/'scores.json').exists()
    cfg=json.loads((root/'frozen_protocol.json').read_text())['config']
    with (root/'postseal_failure_diagnostics.csv').open(encoding='utf-8-sig') as f:diag=list(csv.DictReader(f))
    with (root/'correspondence_scores.csv').open(encoding='utf-8-sig') as f:pairs=list(csv.DictReader(f))
    info={};coverage=[]
    for name in ['test_000','test_001']:
        dest=Path(cfg['dataset'])/name;scene=json.loads((dest/'supervision/scene.json').read_text())
        frames=[load_scene_epoch(cfg['dataset'],name,e,'rich',shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        hist=extract_epoch(frames[0],cfg,cfg['seed']);pose=np.array(scene['poses']);perm=np.array(scene['permutations'])
        for i in range(24):
            visible=[]
            for e in [1,2]:
                G=pose[e,i]@np.linalg.inv(pose[0,i])
                s,_=ray_state(apply(G,hist[i]['xyz']),frames[e],int(perm[e,i]),cfg)
                visible.append(s==1)
            coverage.append(dict(scene=name,asset=i,detail=int(i in scene['detail_assets']),
                history_features=len(hist[i]['xyz']),visible_in_c1=int(visible[0].sum()),visible_in_c2=int(visible[1].sum()),
                visible_in_both=int((visible[0]&visible[1]).sum())))
        rec=json.loads((dest/'frames.json').read_text());pp=json.loads((dest/'supervision/pairs_v3.json').read_text())
        for k,p in enumerate(pp):
            ra,rb=rec[p['source']],rec[p['target']]
            info[name,k]=('detail-detail' if ra['kind']=='detail' and rb['kind']=='detail' else 'overview-overview',ra['epoch'],rb['epoch'])
    pool=[]
    for variant in cfg['variants']:
        rr=[r for r in diag if r['split']=='test' and r['variant']==variant]
        detail={(r['scene'],str(r['asset'])):bool(r['detail']) for r in coverage}
        pool.append(dict(variant=variant,tasks=len(rr),
          closeup_tasks=sum(detail[r['scene'],r['asset']] for r in rr),
          oracle_qualified_with_closeup=sum(int(r['oracle_pool_qualified']) for r in rr if detail[r['scene'],r['asset']]),
          oracle_qualified_without_closeup=sum(int(r['oracle_pool_qualified']) for r in rr if not detail[r['scene'],r['asset']])))
    sampling=[]
    for v in cfg['variants']:
        for m in ['fusion','persistent_visible']:
            for kind in ['detail-detail','overview-overview']:
                rr=[r for r in pairs if r['scene'].startswith('test_') and r['variant']==v and r['method']==m and
                    r['condition']=='visible' and info[r['scene'],int(r['pair'])][0]==kind]
                total=sum(int(r['total']) for r in rr);correct=sum(int(r['correct']) for r in rr)
                sampling.append(dict(variant=v,method=m,kind=kind,visible_points=total,correct=correct,recall=correct/total if total else None))
    save(root/'postseal_coverage_analysis.json',dict(oracle_diagnostic_only=True,source_sha=sha(__file__),
        feature_visibility_uses_gt_pose_and_measured_depth=True,coverage=coverage,pool_coverage=pool,
        correspondence_sampling=sampling,no_prediction_changes=True))
    for r in pool:print('POOL',r)
    for group in [0,1]:
        rr=[r for r in coverage if r['detail']==group]
        print('FEATURE VISIBILITY detail=',group,'n',len(rr),'C2 median',np.median([r['visible_in_c2'] for r in rr]),'both median',np.median([r['visible_in_both'] for r in rr]))
    for r in sampling:
        if r['variant']=='rich':print('SAMPLING',r)


if __name__=='__main__':main()
