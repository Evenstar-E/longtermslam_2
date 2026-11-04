"""Hard/soft/two-state/three-state repairs, fixed equal training budgets."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel
from inspection_decisions.material_correspondence_training_v3 import balanced_loss
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    root=Path(cfg['training_root']);root.mkdir(parents=True,exist_ok=False);cache=Path(cfg['cache_root'])
    seal=json.loads((cache/'seal.json').read_text(encoding='utf-8'));assert seal['train_only'] and not seal['positive_injection']
    records=json.loads((cache/'index.json').read_text(encoding='utf-8'));assert all(r['scene'].startswith('train_') for r in records)
    cfg['appearance_positive_prior']=max(1e-4,sum(r['repainted_covered'] for r in records)/sum(r['states'][2] for r in records))
    for p,h in seal['files'].items():assert sha(cache/p)==h
    sources={p:sha(p) for p in [__file__,'configs/material_correspondence_distribution_v3.json','inspection_decisions/material_correspondence_distribution_v3.py','inspection_decisions/material_correspondence_training_v3.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,sources=sources,cache_sha=sha(cache/'seal.json'),warm_start_sha=sha(cfg['warm_start']),
        primary_seed=31,no_heldout_selection=True,checkpoints='fixed last step'))
    @lru_cache(maxsize=128)
    def group(file):
        with np.load(cache/file) as f:return dict(f)
    warm=torch.load(cfg['warm_start'],map_location='cpu',weights_only=True)['state_dict'];logs=[];start=time.perf_counter()
    runs=[('hard2',31),('soft2',31)]+[('soft3',s) for s in cfg['seeds']]
    for mode,seed in runs:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=CorrespondenceModel(64,cfg['temperature']);model.warm_start(warm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4);window=[]
        for step in range(cfg['steps']):
            rec=records[int(rng.integers(len(records)))];a=group(rec['source']);b=group(rec['target'])
            with np.load(cache/rec['labels']) as f:labels=dict(f)
            ii=np.sort(rng.choice(len(a['patch']),min(cfg['train_queries'],len(a['patch'])),replace=False))
            source=torch.from_numpy(a['patch'][ii]).float();target=torch.from_numpy(b['patch']).float()
            source=torch.rot90(source,int(rng.integers(4)),(-2,-1));target=torch.rot90(target,int(rng.integers(4)),(-2,-1))
            teacher=torch.from_numpy(labels['teacher'][ii]).float();teacher/=teacher.sum(1,keepdim=True).clamp_min(1e-10)
            optimizer.zero_grad(set_to_none=True)
            loss,stat=balanced_loss(model,source,target,torch.from_numpy(b['area']).float(),
                torch.from_numpy(labels['states'][ii]),teacher,torch.from_numpy(labels['changed'][ii]),mode,cfg)
            assert torch.isfinite(loss);loss.backward()
            physical_parameters=[p for name,p in model.named_parameters() if not name.startswith(('appearance.','change.'))]
            appearance_parameters=list(model.appearance.parameters())+list(model.change.parameters())
            torch.nn.utils.clip_grad_norm_(physical_parameters,5.);torch.nn.utils.clip_grad_norm_(appearance_parameters,5.)
            optimizer.step();window.append(stat)
            if (step+1)%100==0:
                row=dict(mode=mode,seed=seed,step=step+1,**{k:float(np.mean([s[k] for s in window])) for k in stat});window=[];logs.append(row)
                save(root/'training_log.json',logs);print('DEPLOYMENT TRAIN',mode,seed,step+1,round(row['loss'],4),'covered',round(row['covered'],1),flush=True)
        torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,mode=mode,steps=cfg['steps']),root/f'{mode}_{seed}.pt')
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'training_seal.json',dict(complete=True,train_only=True,sources=sources,elapsed_s=time.perf_counter()-start,
        warm_start_sha=sha(cfg['warm_start']),cache_sha=sha(cache/'seal.json'),runs=len(runs),
        checkpoints={p.name:sha(p) for p in root.glob('*.pt')}))
    print('DEPLOYMENT TRAINING SEALED',flush=True)


if __name__=='__main__':main()
