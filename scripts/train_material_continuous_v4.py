"""Fixed-budget local surface matcher training; no DEV checkpoint selection."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));cache=Path(v4['cache_root']);seal=json.loads((cache/'seal.json').read_text(encoding='utf-8'));assert seal['complete'] and seal['train_only']
    root=Path(v4['training_root']);root.mkdir(parents=True,exist_ok=False);index=json.loads((cache/'index.json').read_text(encoding='utf-8'));base=Path(seal['anchor_cache'])
    sources={p:sha(p) for p in [__file__,'configs/material_continuous_correspondence_v4.json','inspection_decisions/material_continuous_correspondence_v4.py']}
    @lru_cache(maxsize=None)
    def group(name):
        with np.load(base/name) as f:return f['patch'].astype(np.float32)
    @lru_cache(maxsize=None)
    def label(name):
        with np.load(cache/name) as f:return {k:f[k] for k in f.files}
    catalog=[[],[]]
    for r,record in enumerate(index):
        assert record['scene'].startswith('train_');lab=label(record['labels'])
        for value in [False,True]:
            ii,jj=np.nonzero(lab['positive']==value);catalog[int(value)].extend(zip(np.full(len(ii),r),ii,jj))
    catalog=[np.array(a,int) for a in catalog];save(root/'frozen_protocol.json',dict(config=v4,cache_sha=sha(cache/'seal.json'),class_counts=[len(a) for a in catalog],
        positive_negative_balanced=True,selection='all fixed TRAIN steps; no checkpoint choice',sources=sources))
    start=time.perf_counter()
    for seed in v4['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SurfaceMatcher()
        if seed==31:torch.save(dict(state_dict=model.state_dict(),seed=seed,steps=0),root/'untrained_31.pt')
        opt=torch.optim.AdamW(model.parameters(),lr=v4['learning_rate'],weight_decay=1e-4);logs=[]
        for step in range(v4['steps']):
            batch=np.concatenate([a[rng.integers(len(a),size=v4['batch_size']//2)] for a in catalog]);A=[];B=[];Y=[];M=[]
            for ri,i,k in batch:
                r=index[ri];lab=label(r['labels']);j=int(lab['anchors'][i,k]);A.append(group(r['source'])[i]);B.append(group(r['target'])[j]);M.append(lab['mask'][i,k])
                y=np.zeros(145,np.float32)
                if lab['positive'][i,k]:y[:144]=lab['teacher'][i,k];y/=max(y.sum(),1e-20)
                else:y[-1]=1
                Y.append(y)
            a=torch.from_numpy(np.array(A));b=torch.from_numpy(np.array(B));mask=torch.from_numpy(np.array(M));target=torch.from_numpy(np.array(Y))
            opt.zero_grad(set_to_none=True);score=model(a,b,mask);loss=-(target*score.log_softmax(1)).sum(1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step%100==0 or step==v4['steps']-1:
                logs.append(dict(step=step+1,loss=float(loss.detach())));print('V4 TRAIN',seed,logs[-1],flush=True)
        torch.save(dict(state_dict=model.state_dict(),seed=seed,steps=v4['steps']),root/f'matcher_{seed}.pt');save(root/f'log_{seed}.json',logs)
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'training_seal.json',dict(complete=True,train_only=True,sources=sources,cache_sha=sha(cache/'seal.json'),elapsed_s=time.perf_counter()-start,
        checkpoints={p.name:sha(p) for p in root.glob('*.pt')}))


if __name__=='__main__':main()
