"""Frozen factorial prototype experiment; paired generated scenes across all methods."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from scripts.joint_estimation import simulate,infer,evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/joint_estimation_v1.json');ap.add_argument('--output',default='results/joint_estimation_v1');args=ap.parse_args()
    config=Path(args.config);cfg=json.loads(config.read_text(encoding='utf-8'));out=Path(args.output);out.mkdir(exist_ok=False)
    (out/'frozen_config.json').write_bytes(config.read_bytes());(out/'scenes').mkdir()
    factors=list(cfg['factors']);conditions=[dict(zip(factors,values)) for values in itertools.product(*(cfg['factors'][f] for f in factors))]
    started=time.perf_counter();count=0
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as metrics:
        for cid,condition in enumerate(conditions):
            for seed_index in range(cfg['seeds']):
                sid=cid*cfg['seeds']+seed_index;seed=cfg['seed_base']+sid;obs,truth=simulate(cfg['simulation'],condition,seed)
                payload={**obs,**{'truth_'+k:v for k,v in truth.items()}}
                np.savez_compressed(out/'scenes'/f'{sid:04d}.npz',**payload)
                for method in cfg['methods']:
                    tick=time.perf_counter();pred=infer(obs,cfg['algorithm'],method,seed+400000)
                    row=dict(scene_id=sid,seed=seed,condition=condition,method=method,seconds=time.perf_counter()-tick,**evaluate(pred,truth,cfg['evaluation']))
                    metrics.write(json.dumps(row,allow_nan=False)+'\n');count+=1
                metrics.flush()
            print(json.dumps(dict(condition=cid+1,total=len(conditions),records=count,elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    paths=[config,Path('scripts/joint_estimation.py'),Path(__file__),out/'metrics.jsonl',*sorted((out/'scenes').glob('*.npz'))]
    manifest=dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scene_count=len(conditions)*cfg['seeds'],metric_records=count,seconds=time.perf_counter()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
