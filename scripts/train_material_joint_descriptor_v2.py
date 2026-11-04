"""Train factorization repair using unchanged TRAIN cache, fixed steps and seeds."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='4'
import json
import time
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_joint_descriptor_v2 import FactoredMaterialDescriptor,factored_loss,log_joint_distribution
from scripts.run_textured_material_comparison_v1 import save,sha


def checks():
    torch.manual_seed(8);s=torch.randn(8,9,dtype=torch.double);m=torch.randn(8,dtype=torch.double,requires_grad=True)
    logp=log_joint_distribution(s,m);shift=torch.arange(8,dtype=torch.double)[:,None]*100
    assert torch.max(abs(logp.exp().sum(1)-1))<1e-12
    assert torch.max(abs(logp-log_joint_distribution(s+shift,m)))<1e-12
    assert torch.max(abs(logp[:,:-1].exp().sum(1)-m.sigmoid()))<1e-12
    y=torch.tensor([0,1,2,3,9,9,9,9]);loss=torch.nn.functional.nll_loss(logp,y);loss.backward()
    expected=(m.sigmoid()-torch.tensor([1.,1,1,1,0,0,0,0]))/8
    assert torch.max(abs(m.grad-expected))<1e-12
    print('PASS normalized probabilities, score-shift invariance, visibility marginal and analytic gradient',flush=True)


def main():
    checks();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text(encoding='utf-8'))
    root=Path('results/material_joint_descriptor_training_v2');root.mkdir(parents=True,exist_ok=False)
    cache=Path(cfg['cache_root']);seal=json.loads((cache/'seal.json').read_text(encoding='utf-8'));assert seal['split']=='train'
    records=json.loads((cache/'index.json').read_text(encoding='utf-8'));assert all(r['scene'].startswith('train_') for r in records)
    prior=1-seal['totals']['repaint_visible']/seal['totals']['visible']
    sources={p:sha(p) for p in [__file__,'configs/material_joint_descriptor_v1.json',
        'inspection_decisions/material_joint_descriptor_v1.py','inspection_decisions/material_joint_descriptor_v2.py']}
    save(root/'frozen_training_protocol.json',dict(config=cfg,sources=sources,cache_seal_sha=sha(cache/'seal.json'),
        visible_persistence_prior=prior,reason='Training-only V1 probability semantic mismatch and class imbalance',
        no_heldout_labels=True,checkpoint='Fixed last step'))
    histories=[];start=time.perf_counter()
    for mode,seed,balanced in [('factorized',31,False)]+[('balanced',s,True) for s in cfg['training_seeds']]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=FactoredMaterialDescriptor(cfg['embedding_dim'],prior,balanced)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4);window=[]
        for step in range(cfg['steps']):
            rec=records[int(rng.integers(len(records)))]
            with np.load(cache/rec['file']) as f:arr={k:torch.from_numpy(f[k]).float() if k in ['source','target'] else torch.from_numpy(f[k]) for k in f.files}
            arr['source']=torch.rot90(arr['source'],int(rng.integers(4)),(-2,-1));arr['target']=torch.rot90(arr['target'],int(rng.integers(4)),(-2,-1))
            optimizer.zero_grad(set_to_none=True)
            loss,stats=factored_loss(model,arr['source'],arr['target'],arr['visible'],arr['positive'],arr['changed'],arr['near'],cfg['temperature'],cfg['head_weight'])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();window.append(stats)
            if (step+1)%50==0:
                r=dict(mode=mode,seed=seed,step=step+1,**{k:float(np.mean([x[k] for x in window])) for k in stats})
                histories.append(r);window=[];save(root/'training_log.json',histories)
                print('FACTOR TRAIN',mode,seed,step+1,round(r['loss'],4),round(r['retrieval'],4),flush=True)
        torch.save(dict(state_dict=model.state_dict(),config=cfg,seed=seed,mode=mode,balanced=balanced,
            persistence_prior=prior,steps=cfg['steps']),root/f'{mode}_{seed}.pt')
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'training_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,
        elapsed_s=time.perf_counter()-start,cache_seal_sha=sha(cache/'seal.json'),persistence_prior=prior,
        checkpoints={p.name:sha(p) for p in root.glob('*.pt')},training_runs=4))
    print('FACTORIZED TRAINING SEALED',flush=True)


if __name__=='__main__':main()
