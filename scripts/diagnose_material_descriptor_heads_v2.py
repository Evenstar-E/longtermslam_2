"""Calibration of factored heads on TRAIN only, after fixed-step training."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_joint_descriptor_v2 import FactoredMaterialDescriptor
from scripts.diagnose_material_descriptor_training_v1 import auc
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(2);cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'))
    root=Path('results/material_joint_descriptor_training_v2');cache=Path(cfg['cache_root']);index=json.loads((cache/'index.json').read_text(encoding='utf-8'));result=[]
    for mode,seed in [('factorized',31),('balanced',31),('balanced',47),('balanced',73)]:
        ck=torch.load(root/f'{mode}_{seed}.pt',map_location='cpu',weights_only=True)
        model=FactoredMaterialDescriptor(64,ck['persistence_prior'],ck['balanced']);model.load_state_dict(ck['state_dict']);model.eval()
        vis=[];mp=[];changed=[];cp=[];correct=0;total=0
        with torch.inference_mode():
            for rec in index:
                assert rec['scene'].startswith('train_')
                with np.load(cache/rec['file']) as f:a=model(torch.from_numpy(f['source']).float());b=model(torch.from_numpy(f['target']).float());v=f['visible'];positive=f['positive'];change=f['changed']
                vis.extend(v);mp.extend(model.context_logits(a,b).sigmoid().numpy());ii=np.flatnonzero(v)
                correct+=int(((a@b.T).argmax(1).numpy()[ii]==positive[ii]).sum());total+=len(ii)
                changed.extend(change[ii]);cp.extend(model.persistence_logits(a[ii],b[positive[ii]]).sigmoid().numpy())
        vis=np.array(vis);mp=np.array(mp);changed=np.array(changed);cp=np.array(cp)
        r=dict(mode=mode,seed=seed,training_only=True,retrieval=correct/total,
            matchability_auc=auc(vis,mp),matchability_brier=float(np.mean((vis-mp)**2)),
            mean_matchability_visible=float(mp[vis].mean()),mean_matchability_hidden=float(mp[~vis].mean()),
            persistence_auc=auc(~changed,cp),mean_persistence_changed=float(cp[changed].mean()),
            mean_persistence_unchanged=float(cp[~changed].mean()),repaint_detected_at_05=int((cp[changed]<.5).sum()),
            repaint_total=int(changed.sum()),unchanged_false_repaint_at_05=int((cp[~changed]<.5).sum()))
        result.append(r);print('FACTOR CALIBRATION',r,flush=True)
    save(root/'training_calibration.json',dict(training_only=True,no_heldout_labels=True,results=result,source_sha=sha(__file__)))


if __name__=='__main__':main()
