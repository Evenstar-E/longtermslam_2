"""Fixed-step supervised CPU training; no held-out labels or checkpoint selection."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor,training_loss
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/material_joint_descriptor_v1.json');args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    root=Path(cfg['training_root'])
    if root.exists() and any(root.iterdir()):raise FileExistsError('Refuse nonempty training root')
    root.mkdir(parents=True,exist_ok=True)
    cache=Path(cfg['cache_root']);seal=json.loads((cache/'seal.json').read_text(encoding='utf-8'));assert seal['split']=='train' and seal['complete']
    records=json.loads((cache/'index.json').read_text(encoding='utf-8'))
    assert all(r['scene'].startswith('train_') for r in records)
    for f,h in seal['files'].items():assert sha(cache/f)==h
    sources={p:sha(p) for p in [args.config,__file__,'inspection_decisions/material_joint_descriptor_v1.py']}
    save(root/'frozen_training_protocol.json',dict(config=cfg,sources=sources,cache_seal_sha=sha(cache/'seal.json'),
        device='cpu',torch=torch.__version__,checkpoints='last fixed step only',heldout_labels_read=False))
    histories=[];start=time.perf_counter()
    runs=[('contrastive',cfg['training_seeds'][0],0.)]+[('full',s,cfg['head_weight']) for s in cfg['training_seeds']]
    for mode,seed,aux in runs:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=MaterialDescriptor(cfg['embedding_dim'])
        if mode=='contrastive':torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,mode='untrained'),root/'untrained.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
        runstart=time.perf_counter();window=[]
        for step in range(cfg['steps']):
            rec=records[int(rng.integers(len(records)))]
            with np.load(cache/rec['file']) as f:arr={k:torch.from_numpy(f[k]).float() if k in ['source','target'] else torch.from_numpy(f[k]) for k in f.files}
            # Independent quarter-turns enforce robustness to tangent-frame orientation.
            arr['source']=torch.rot90(arr['source'],int(rng.integers(4)),(-2,-1))
            arr['target']=torch.rot90(arr['target'],int(rng.integers(4)),(-2,-1))
            optimizer.zero_grad(set_to_none=True)
            loss,stats=training_loss(model,arr['source'],arr['target'],arr['visible'],arr['positive'],arr['changed'],arr['near'],cfg['temperature'],aux)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            window.append(stats)
            if (step+1)%50==0:
                report=dict(mode=mode,seed=seed,step=step+1,elapsed_s=time.perf_counter()-runstart,
                    **{k:float(np.mean([r[k] for r in window])) for k in stats})
                histories.append(report);window=[];save(root/'training_log.json',histories)
                print('TRAIN',mode,seed,step+1,'loss',round(report['loss'],4),'retrieval',round(report['retrieval'],4),round(report['elapsed_s'],1),flush=True)
        path=root/f'{mode}_{seed}.pt'
        torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,mode=mode,steps=cfg['steps']),path)
        assert all(sha(p)==h for p,h in sources.items())
    save(root/'training_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,
        elapsed_s=time.perf_counter()-start,cache_seal_sha=sha(cache/'seal.json'),
        checkpoints={p.name:sha(p) for p in root.glob('*.pt')},train_records=len(records),training_runs=len(runs)))
    print('TRAINING SEALED',len(runs),'runs',flush=True)


if __name__=='__main__':main()
