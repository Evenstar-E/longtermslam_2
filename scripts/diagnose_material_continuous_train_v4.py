"""Fixed TRAIN replay only; reports fit, never selects model/threshold/steps."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));root=Path(v4['training_root']);cache=Path(v4['cache_root'])
    seal=json.loads((cache/'seal.json').read_text(encoding='utf-8'));base=Path(seal['anchor_cache']);index=json.loads((cache/'index.json').read_text(encoding='utf-8'))
    selected=np.linspace(0,len(index)-1,160,dtype=int);A=[];B=[];Y=[];M=[];covered=[]
    for ri in selected:
        r=index[ri];assert r['scene'].startswith('train_')
        with np.load(base/r['source']) as f:a=f['patch']
        with np.load(base/r['target']) as f:b=f['patch']
        with np.load(cache/r['labels']) as f:l={k:f[k] for k in f.files}
        for pos in [False,True]:
            ii,jj=np.nonzero(l['positive']==pos)
            for k in np.linspace(0,len(ii)-1,min(8,len(ii)),dtype=int):
                i,j=ii[k],jj[k];A.append(a[i]);B.append(b[l['anchors'][i,j]]);Y.append(l['teacher'][i,j]);M.append(l['mask'][i,j]);covered.append(pos)
    a=torch.from_numpy(np.array(A,dtype=np.float32));b=torch.from_numpy(np.array(B,dtype=np.float32));mask=torch.from_numpy(np.array(M));y=np.array(Y);positive=np.array(covered);results=[]
    with torch.inference_mode():
        for file in ['untrained_31.pt','matcher_31.pt','matcher_47.pt']:
            model=SurfaceMatcher();model.load_state_dict(torch.load(root/file,weights_only=True,map_location='cpu')['state_dict']);model.eval();values=[]
            for start in range(0,len(a),256):values.append(model(a[start:start+256],b[start:start+256],mask[start:start+256]).numpy())
            s=np.concatenate(values);pred=s.argmax(1);local=s[:,:144].argmax(1);soft=np.exp(s[:,:144]-s[:,:144].max(1,keepdims=True));soft*=np.array(M);soft/=np.maximum(soft.sum(1,keepdims=True),1e-12)
            nonnull=1/(1+np.exp(np.clip(s[:,144]-np.logaddexp.reduce(s[:,:144],axis=1),-60,60)))
            results.append(dict(model=file,positive=int(positive.sum()),negative=int((~positive).sum()),
                positive_argmax_in_teacher_support=float((y[np.arange(len(y)),local]>0)[positive].mean()),
                positive_predicted_nonnull=float((pred[positive]!=144).mean()),negative_predicted_nonnull=float((pred[~positive]!=144).mean()),
                positive_nonnull_marginal_mean=float(nonnull[positive].mean()),negative_nonnull_marginal_mean=float(nonnull[~positive].mean()),
                positive_nonnull_marginal_above_half=float((nonnull[positive]>.5).mean()),negative_nonnull_marginal_above_half=float((nonnull[~positive]>.5).mean()),
                mean_teacher_mass_at_top1=float(y[np.arange(len(y)),local][positive].mean())))
    save(root/'train_replay_diagnostic.json',dict(source_sha=sha(__file__),TRAIN_replay=True,independent_generalization=False,selection='160 evenly spaced TRAIN records, up to8 positive and8 negative pairs each',results=results))
    print('V4 TRAIN REPLAY',results,flush=True)


if __name__=='__main__':main()
