"""Training-only descriptive calibration, not held-out performance."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor
from scripts.run_textured_material_comparison_v1 import save,sha


def auc(y,p):
    y=np.array(y,bool);p=np.array(p);n=y.sum();m=len(y)-n
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main():
    torch.set_num_threads(2);cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'))
    root=Path(cfg['training_root']);cache=Path(cfg['cache_root']);index=json.loads((cache/'index.json').read_text(encoding='utf-8'))
    result=[]
    for mode,seed in [('contrastive',31),('full',31),('full',47),('full',73)]:
        model=MaterialDescriptor(cfg['embedding_dim']);model.load_state_dict(torch.load(root/f'{mode}_{seed}.pt',weights_only=True)['state_dict']);model.eval()
        vis=[];mp=[];changed=[];cp=[];correct=0;total=0
        with torch.inference_mode():
            for rec in index:
                assert rec['scene'].startswith('train_')
                with np.load(cache/rec['file']) as f:a=model(torch.from_numpy(f['source']).float());b=model(torch.from_numpy(f['target']).float());v=f['visible'];positive=f['positive'];change=f['changed']
                mm=model.context_logits(a,b).sigmoid().numpy();vis.extend(v);mp.extend(mm)
                ii=np.flatnonzero(v);correct+=int(((a@b.T).argmax(1).numpy()[ii]==positive[ii]).sum());total+=len(ii)
                persist=model.persistence_logits(a[ii],b[positive[ii]]).sigmoid().numpy()
                changed.extend(change[ii]);cp.extend(persist)
        vis=np.array(vis);mp=np.array(mp);changed=np.array(changed);cp=np.array(cp)
        r=dict(mode=mode,seed=seed,training_only=True,visible_queries=total,retrieval=correct/total,
            matchability_auc=auc(vis,mp),matchability_brier=float(np.mean((vis-mp)**2)),
            mean_matchability_visible=float(mp[vis].mean()),mean_matchability_hidden=float(mp[~vis].mean()),
            visible_repaint_points=int(changed.sum()),persistence_auc=auc(~changed,cp),
            mean_persistence_changed=float(cp[changed].mean()),mean_persistence_unchanged=float(cp[~changed].mean()),
            repaint_detected_at_05=int((cp[changed]<.5).sum()))
        result.append(r);print('TRAIN CALIBRATION',r,flush=True)
    save(root/'training_calibration.json',dict(training_only=True,no_heldout_labels=True,results=result,source_sha=sha(__file__)))


if __name__=='__main__':main()
