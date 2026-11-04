"""Post-seal oracle diagnostics; never write predictions or tune parameters."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_matching_v1 import apply,extract_epoch,geometric_pools,matches,ransac,icp
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/textured_material_comparison_v1');args=ap.parse_args()
    root=Path(args.root);assert (root/'prediction_seal.json').exists() and (root/'scores.json').exists()
    protocol=json.loads((root/'frozen_protocol.json').read_text());cfg=protocol['config'];rows=[]
    for name in protocol['scenes']:
        dest=Path(cfg['dataset'])/name;truth=json.loads((dest/'supervision/scene.json').read_text())
        pose=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(dest)
        gp=None
        for variant in cfg['variants']:
            frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
            epochs=[extract_epoch(f,cfg,cfg['seed']+e) for e,f in enumerate(frames)]
            if gp is None:gp=geometric_pools(epochs,cfg)
            pred={}
            for m in ['geometry','texture','fusion','persistent_visible']:
                with np.load(root/name/f'{variant}_{m}.npz') as f:pred[m]=dict(f)
            for i in range(24):
                pp=[];nin=[];nmatch=[];nfeatures=[];ms=[]
                for e in [1,2]:
                    j=int(perm[e,i]);a,b=epochs[0][i],epochs[e][j]
                    mm=matches(a,b,cfg);ms.append(mm)
                    nmatch.append(len(mm['ia']));nfeatures.append(len(b['desc']))
                    T=ransac(a,b,mm,cfg,cfg['seed']+e*1000+i*24+j)
                    pool=list(gp[0,e][i][j])
                    if T is not None:pool += [T,icp(a,b,T,cfg)]
                    pp.append(pool)
                    if len(mm['ia']):
                        qa=apply(np.linalg.inv(pose[0,i]),a['xyz'][mm['ia']])
                        qb=apply(np.linalg.inv(pose[e,i]),b['xyz'][mm['ib']])
                        nin.append(int((np.linalg.norm(qa-qb,axis=1)<=cfg['material_scale_m']).sum()))
                    else:nin.append(0)
                q=anchors[i];x=apply(pose[0,i],q);y=[apply(pose[e,i],q) for e in [1,2]]
                best=np.inf
                for T1 in pp[0]:
                    for T2 in pp[1]:
                        z1,z2=apply(T1,x),apply(T2,x)
                        errs=[np.quantile(np.linalg.norm(z1-y[0],axis=1),.95),
                              np.quantile(np.linalg.norm(z2-y[1],axis=1),.95),
                              np.quantile(np.linalg.norm((z2-z1)-(y[1]-y[0]),axis=1),.95)]
                        best=min(best,max(errs))
                row=dict(scene=name,split=name.rsplit('_',1)[0],variant=variant,asset=i,
                    clone_pair=int(i in [22,23]),changed=int(i in truth['changed_assets']),
                    true_h1_matches=nmatch[0],true_h2_matches=nmatch[1],true_h1_material_inliers=nin[0],
                    true_h2_material_inliers=nin[1],historical_features=len(epochs[0][i]['desc']),
                    current1_features=nfeatures[0],current2_features=nfeatures[1],
                    shared_historical_descriptor_indices=len(np.intersect1d(ms[0]['ia'],ms[1]['ia'])),
                    oracle_true_id_common_pool_worst_error_m=float(best),oracle_pool_qualified=int(best<=cfg['joint_success_m']))
                for m,p in pred.items():
                    row[m+'_both_ids']=int(np.array_equal(p['target_slots'][i],perm[1:,i]))
                rows.append(row)
        print('DIAGNOSED',name,flush=True)
    write_csv(root/'postseal_failure_diagnostics.csv',rows)
    save(root/'postseal_diagnostic_manifest.json',dict(oracle_only=True,no_predictions_modified=True,
        source_sha=sha(__file__),prediction_seal_sha=sha(root/'prediction_seal.json'),
        table_sha=sha(root/'postseal_failure_diagnostics.csv'),rows=len(rows)))


if __name__=='__main__':main()
