"""Independent balanced repaint curriculum; physical model byte-identical."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import hashlib
import json
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from scipy.stats import rankdata
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel
from scripts.run_textured_material_comparison_v1 import save,sha


def physical_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if not name.startswith(('appearance.','change.')):h.update(name.encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def auc(y,p):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'))
    root=Path('results/material_appearance_training_v3');root.mkdir(parents=True,exist_ok=False)
    cache=Path(cfg['cache_root']);index=json.loads((cache/'index.json').read_text(encoding='utf-8'));index=[r for r in index if r['true_candidate']]
    @lru_cache(maxsize=128)
    def group(file):
        with np.load(cache/file) as f:return f['patch'][:,:6]
    count=sum(r['states'][2] for r in index);A=np.empty((count,6,12,12),np.float16);B=np.empty_like(A);Y=np.zeros(count,bool);cursor=0
    for r in index:
        assert r['scene'].startswith('train_')
        with np.load(cache/r['labels']) as f:ii=np.flatnonzero(f['states']==2);jj=f['teacher'][ii].argmax(1);changed=f['changed'][ii]
        n=len(ii);A[cursor:cursor+n]=group(r['source'])[ii];B[cursor:cursor+n]=group(r['target'])[jj];Y[cursor:cursor+n]=changed;cursor+=n
    pos=np.flatnonzero(Y);neg=np.flatnonzero(~Y);assert len(pos)>0
    save(root/'frozen_protocol.json',dict(steps=300,batch=48,balanced_point_pairs=True,train_only=True,
        input_cache_sha=sha(cache/'seal.json'),covered_pairs=count,repainted_pairs=len(pos),
        physical_weights_frozen=True,fpr_target=cfg['appearance_train_fpr'],source_sha=sha(__file__)))
    results=[];start=time.perf_counter()
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        ck=torch.load(Path(cfg['training_root'])/f'soft3_{seed}.pt',weights_only=True,map_location='cpu')
        model=CorrespondenceModel();model.load_state_dict(ck['state_dict']);initial=physical_hash(model)
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('appearance.','change.')))
        params=list(model.appearance.parameters())+list(model.change.parameters());opt=torch.optim.AdamW(params,lr=.0005,weight_decay=1e-4)
        for step in range(300):
            ii=np.r_[rng.choice(pos,24,replace=True),rng.choice(neg,24,replace=True)]
            a=torch.from_numpy(A[ii]).float();b=torch.from_numpy(B[ii]).float()
            a=torch.rot90(a,int(rng.integers(4)),(-2,-1));b=torch.rot90(b,int(rng.integers(4)),(-2,-1))
            opt.zero_grad(set_to_none=True);score=model.change_logits(model.encode_appearance(a),model.encode_appearance(b))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(score,torch.from_numpy(Y[ii]).float());loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);opt.step()
        scores=[]
        with torch.inference_mode():
            for k in range(0,count,512):
                a=model.encode_appearance(torch.from_numpy(A[k:k+512]).float());b=model.encode_appearance(torch.from_numpy(B[k:k+512]).float())
                scores.extend(model.change_logits(a,b).sigmoid().numpy())
        scores=np.array(scores);threshold=float(np.quantile(scores[~Y],1-cfg['appearance_train_fpr']))
        assert physical_hash(model)==initial
        checkpoint={**ck,'state_dict':model.state_dict(),'appearance_steps':300,'appearance_threshold':threshold,'physical_hash':initial}
        torch.save(checkpoint,root/f'soft3_{seed}.pt')
        result=dict(seed=seed,train_auc=auc(Y,scores),threshold=threshold,
            train_repaint_recall=float((scores[Y]>threshold).mean()),train_unchanged_fpr=float((scores[~Y]>threshold).mean()),physical_hash=initial)
        results.append(result);print('APPEARANCE REPAIR',result,flush=True)
    save(root/'training_seal.json',dict(complete=True,train_only=True,source_sha=sha(__file__),
        physical_model_unchanged=True,elapsed_s=time.perf_counter()-start,calibration=results,
        checkpoints={p.name:sha(p) for p in root.glob('*.pt')}))
    print('APPEARANCE REPAIR SEALED',flush=True)


if __name__=='__main__':main()
