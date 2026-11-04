"""Post-seal only: did finite top-mode propagation omit existing good modes?"""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def batch_error(T,x,y1,y2):
    result=[]
    for begin in range(0,len(T),128):
        U=T[begin:begin+128];a=np.einsum('bij,nj->bni',U[:,0,:3,:3],x)+U[:,0,None,:3,3];b=np.einsum('bij,nj->bni',U[:,1,:3,:3],x)+U[:,1,None,:3,3]
        residual=np.stack([a-y1,b-y2,(b-a)-(y2-y1)],1)
        result.append(np.quantile(np.linalg.norm(residual,axis=-1),.95,axis=-1).max(1))
    return np.concatenate(result)


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);rows=[]
    import csv
    stages=list(csv.DictReader((root/'oracle_by_stage.csv').open(encoding='utf-8-sig')));stage_lookup={(r['variant'],int(r['asset'])):r for r in stages}
    for v in cfg['variants']:
        with np.load(Path(cfg['parent_root'])/f'{v}_pools.npz') as f:pools=dict(f)
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]];C=pools['transforms'][2,j,k,:pools['counts'][2,j,k]]
            x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
            a=np.repeat(A,len(C),0);c=np.tile(C,(len(A),1,1));forward=np.stack([a,c@a],1)
            b=np.repeat(B,len(C),0);c=np.tile(C,(len(B),1,1));backward=np.stack([inverse(c)@b,b],1)
            ef=batch_error(forward,x,y1,y2);eb=batch_error(backward,x,y1,y2);best=min(float(ef.min()),float(eb.min()))
            retained_union=min(best,float(stage_lookup[v,i]['retained_error_m']))
            rows.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),all_forward_pairs=len(forward),all_backward_pairs=len(backward),full_forward_error_m=float(ef.min()),full_backward_error_m=float(eb.min()),full_propagation_error_m=best,full_propagation_qualified=int(best<=.05),full_propagation_with_retained_error_m=retained_union,full_propagation_with_retained_qualified=int(retained_union<=.05)))
    write_csv(root/'full_propagation_oracle.csv',rows);summary=[]
    for v in cfg['variants']:
        rr=[r for r in rows if r['variant']==v];old={int(r['asset']):r for r in stages if r['variant']==v}
        summary.append(dict(variant=v,full_propagation_qualified=sum(r['full_propagation_qualified'] for r in rr),full_propagation_with_retained_qualified=sum(r['full_propagation_with_retained_qualified'] for r in rr),full_rear_qualified=sum(r['full_propagation_qualified'] for r in rr if r['rear']),budget_missed_qualified=[r['asset'] for r in rr if r['full_propagation_qualified'] and float(old[r['asset']]['propagated_error_m'])>.05],scope='Private diagnostic only: all existing A/C or B/C poses, NOT inferred task success or a higher-budget baseline prediction. with_retained adds exactly the same old pool as the finite-budget comparison.'))
    save(root/'full_propagation_summary.json',dict(summary=summary,rows=len(rows),private_only_after_seal=True));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
