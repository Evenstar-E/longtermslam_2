"""Four actual event-likelihood policies; all predictions precede evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v7 import run_selection_v7
from comparative_evidence.evaluation_v3 import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=2);args=parser.parse_args()
    assert 0<=args.shard<args.shards
    cfgpath=Path('configs/comparative_selection_v7.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/comparative_selection_v7')/f'shard_{args.shard}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'predictions').mkdir();(out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(__file__)]+list(Path('comparative_evidence').glob('*.py'))
    manifest=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        executions=0,evaluations=0,exclusions=[],shard=args.shard,shards=args.shards)
    begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(args.shard,cfg['scenes'],args.shards):
            root=Path(cfg['source_data'])/f'{sid:04d}'
            for method in cfg['methods']:
                result=execute_policy(root,*cfg['budget'],lambda port:run_selection_v7(port,method))
                p=result['prediction'];target=out/'predictions'/f'{sid:04d}_{method}.json'
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
                for name,pred in {'risk':p,**p['terminal_alternatives']}.items():
                    f.write(json.dumps(dict(scene=sid,method=method,variant=name,**evaluate(root,pred),
                        cost=result['cost'],timings=result['timings'],prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))+'\n')
                    manifest['evaluations']+=1
                f.flush();manifest['executions']+=1
            (out/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'active shard{args.shard} scene{sid:02d} {manifest["executions"]} policies {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
