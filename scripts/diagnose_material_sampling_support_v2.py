"""Post-seal oracle links: distinguish sampled support from descriptor failures."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.material_feature_comparison_v1 import measured_epoch
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    root=Path('results/material_joint_descriptor_comparison_v2');assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'));data=Path(cfg['dataset']);rows=[]
    for name in json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'))['scenes']:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));pose=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(dest)
        epochs=[]
        for e in range(3):
            frames=load_scene_epoch(data,name,e,'flat',shuffle_seed=cfg['seed']+e)[0]
            pieces=measured_epoch(frames,cfg);slots=[]
            for slot in range(24):
                vv=[p['xyz'][p['slots']==slot] for p in pieces if (p['slots']==slot).any()]
                slots.append(np.concatenate(vv) if vv else np.zeros((0,3)))
            epochs.append(slots)
        for i in range(24):
            x=epochs[0][i];H=np.linalg.inv(pose[0,i]);qh=apply(H,x);transforms=[];supports=[]
            for e in [1,2]:
                y=epochs[e][perm[e,i]];qy=apply(np.linalg.inv(pose[e,i]),y)
                d,ix=cKDTree(qy).query(qh);take=d<=.025;supports.append(int(take.sum()))
                transforms.append(rigid(x[take],y[ix[take]]))
            worst=1e6
            if all(T is not None for T in transforms):
                q=anchors[i];xx=apply(pose[0,i],q);z=[apply(T,xx) for T in transforms];actual=[apply(pose[e,i],q) for e in [1,2]]
                worst=max(np.quantile(np.linalg.norm(z[0]-actual[0],axis=1),.95),np.quantile(np.linalg.norm(z[1]-actual[1],axis=1),.95),
                    np.quantile(np.linalg.norm((z[1]-z[0])-(actual[1]-actual[0]),axis=1),.95))
            rows.append(dict(scene=name,asset=i,detail=int(i in truth['detail_assets']),sampled_h1_support=supports[0],sampled_h2_support=supports[1],
                oracle_material_links_worst_error_m=float(worst),oracle_material_links_qualified=int(worst<=.05)))
        print('SAMPLING ORACLE',name,flush=True)
    write_csv(root/'postseal_sampling_oracle.csv',rows)
    groups=[]
    for detail in [0,1]:
        rr=[r for r in rows if r['detail']==detail]
        groups.append(dict(detail=detail,tasks=len(rr),oracle_qualified=sum(r['oracle_material_links_qualified'] for r in rr),
            median_h1_sample_support=float(np.median([r['sampled_h1_support'] for r in rr])),
            median_h2_sample_support=float(np.median([r['sampled_h2_support'] for r in rr]))))
    save(root/'postseal_sampling_oracle.json',dict(oracle_only=True,not_algorithm_accuracy=True,source_sha=sha(__file__),
        qualification=sum(r['oracle_material_links_qualified'] for r in rows),tasks=len(rows),groups=groups))
    print('SAMPLING ORACLE RESULT',groups,flush=True)


if __name__=='__main__':main()
