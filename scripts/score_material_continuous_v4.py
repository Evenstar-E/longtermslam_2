"""Post-seal full task scoring, own-family oracle and exact candidate retention."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_correspondence_inference_v3 import inverse
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_joint_descriptor_comparison_v1 import aggregate
from scripts.run_textured_material_comparison_v1 import save,sha


def oracle_fast(truth,pool,anchors):
    poses=np.array(truth['poses']);perm=np.array(truth['permutations']);out=[]
    for i,q in enumerate(anchors):
        x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
        A=pool['transforms'][0,i,perm[1,i],:pool['counts'][0,i,perm[1,i]]]
        B=pool['transforms'][1,i,perm[2,i],:pool['counts'][1,i,perm[2,i]]]
        z1=np.einsum('bij,nj->bni',A[:,:3,:3],x)+A[:,None,:3,3];z2=np.einsum('bij,nj->bni',B[:,:3,:3],x)+B[:,None,:3,3]
        e1=np.quantile(np.linalg.norm(z1-y1,axis=2),.95,axis=1);e2=np.quantile(np.linalg.norm(z2-y2,axis=2),.95,axis=1)
        e12=np.quantile(np.linalg.norm((z2[None]-z1[:,None])-(y2-y1),axis=3),.95,axis=2)
        out.append(float(np.maximum(np.maximum(e1[:,None],e2[None]),e12).min()))
    return out


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));root=Path(v4['comparison_root'])
    proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];checks=1
    # Mandatory prescore addendum. Also supports the already-running original
    # finish process, which imports this scorer only after the main seal exists.
    from scripts.run_material_continuous_retention_control_v4 import CONTROL_ROOT,main as retained_control
    if not (CONTROL_ROOT/'prediction_seal.json').exists():retained_control()
    cs=json.loads((CONTROL_ROOT/'prediction_seal.json').read_text(encoding='utf-8'));assert cs['complete'] and not cs['heldout_labels_read'];checks+=1
    for p,h in cs['sources'].items():assert sha(p)==h;checks+=1
    for p,h in cs['files'].items():assert sha(CONTROL_ROOT/p)==h;checks+=1
    methods=['baseline','retained']+[m for m in proto['methods'] if m!='baseline']
    controls=proto['paired_controls']+[dict(baseline='baseline',candidate='retained')]+[dict(baseline='retained',candidate=m) for m in proto['methods'] if m!='baseline']
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    data=Path(proto['config']['dataset']);assert sha(data/'release_manifest.json')==proto['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    rows=[]
    for name in proto['scenes']:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(dest)
        split=json.loads((dest/'public_metadata.json').read_text(encoding='utf-8'))['split']
        for v in proto['config']['variants']:
            for m in methods:
                active=CONTROL_ROOT if m=='retained' else root
                with np.load(active/name/f'{v}_{m}_family_pools.npz') as f:pool=dict(f)
                oracle=oracle_fast(truth,pool,anchors)
                # Every family must retain all six old complete hypotheses and
                # exact V3 strong baseline; no good candidates silently lost.
                refs=[]
                for oldm in ['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
                    with np.load(Path('results/textured_material_comparison_v1')/name/f'{v}_{oldm}.npz') as f:refs.append(dict(f))
                with np.load(Path(proto['config']['comparison_root'])/name/f'{v}_ray.npz') as f:refs.append(dict(f))
                for pred in refs:
                    for i,(j,k) in enumerate(pred['target_slots']):
                        for pe,a,b,T in [(0,i,j,pred['transforms'][i,0]),(1,i,k,pred['transforms'][i,1]),(2,j,k,pred['transforms'][i,1]@inverse(pred['transforms'][i,0]))]:
                            values=pool['transforms'][pe,a,b,:pool['counts'][pe,a,b]]
                            assert np.min(np.max(abs(values-T),axis=(1,2)))<1e-7;checks+=1
                with np.load(active/name/f'{v}_{m}.npz') as f:pred=dict(f)
                for e in [0,1]:assert np.array_equal(np.sort(pred['target_slots'][:,e]),np.arange(24));checks+=1
                R=pred['transforms'][:,:,:3,:3];assert np.isfinite(pred['transforms']).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
                if m=='baseline':assert sha(root/name/f'{v}_{m}.npz')==sha(Path(proto['config']['comparison_root'])/name/f'{v}_ray.npz');checks+=1
                rr=task_rows(name,split,v,m,pred,truth,anchors,proto['base_config'])
                for i,r in enumerate(rr):r.update(oracle_pool_worst_error_m=oracle[i],oracle_pool_qualified=int(oracle[i]<=.05),detail=int(i in truth['detail_assets']))
                rows+=rr
        print('V4 SCORED',name,flush=True)
    write_csv(root/'task_scores.csv',rows)
    result=aggregate(rows,[],methods,proto['config']['variants'],controls);save(root/'scores.json',result)
    save(root/'scoring_audit.json',dict(checks=checks,all_predictions_sealed_before_truth=True,old_and_v3_poses_retained_in_every_family=True,
        task_rows=len(rows),source_sha=sha(__file__),prediction_seal_sha=sha(root/'prediction_seal.json'),
        prescore_control_seal_sha=sha(CONTROL_ROOT/'prediction_seal.json'),prescore_control_completed_before_truth=True,
        hashes={p:sha(root/p) for p in ['scores.json','task_scores.csv']}))
    print('V4 SCORE SEALED',checks,len(rows),flush=True)


if __name__=='__main__':main()
