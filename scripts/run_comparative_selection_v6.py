"""Two new weight-ablated policies and two exact v5 diagnostic replays."""
import argparse,hashlib,json,time
from pathlib import Path
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v6 import run_selection_v6
from comparative_evidence.evaluation_v3 import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
    assert 0<=args.shard<args.shards
    cfgpath=Path('configs/comparative_selection_v6.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/comparative_selection_v6')/f'shard_{args.shard}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'predictions').mkdir();(out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(__file__)]+list(Path('comparative_evidence').glob('*.py'))
    manifest=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        inputs={},shard=args.shard,shards=args.shards,executions=0,new_policies=0,replays=0,
        terminal_evaluations=0,snapshot_evaluations=0,exclusions=[])
    begin=time.perf_counter();methods=cfg['new_methods']+list(cfg['replay_methods'])
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f, (out/'steps.jsonl').open('w',encoding='utf-8') as g:
        for sid in range(args.shard,cfg['scenes'],args.shards):
            root=Path(cfg['source_data'])/f'{sid:04d}'
            for method in methods:
                original=None;trace=None
                if method in cfg['replay_methods']:
                    oldname=cfg['replay_methods'][method]
                    path=Path(f'results/comparative_selection_v5/shard_{sid%2}/predictions/{sid:04d}_{oldname}.json')
                    raw=path.read_bytes();manifest['inputs'][str(path)]=hashlib.sha256(raw).hexdigest()
                    original=json.loads(raw);trace=original['prediction']['decisions']
                result=execute_policy(root,*cfg['budget'],lambda port:run_selection_v6(port,method,trace))
                p=result['prediction']
                if original is not None:
                    assert p['assets']==original['prediction']['assets']
                    manifest['replays']+=1
                else:manifest['new_policies']+=1
                target=out/'predictions'/f'{sid:04d}_{method}.json'
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
                for name,pred in {'risk':p,**p['terminal_alternatives']}.items():
                    f.write(json.dumps(dict(scene=sid,method=method,variant=name,**evaluate(root,pred),
                        cost=result['cost'],prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))+'\n')
                    manifest['terminal_evaluations']+=1
                # Evaluation block AFTER inference. No truth was available to scoring.
                for stage,s in enumerate(p['snapshots']):
                    g.write(json.dumps(dict(scene=sid,method=method,stage=stage,
                        model_risk32=s['model_risk32'],terminal_risk2048=s['terminal_risk2048'],
                        risk=evaluate(root,s['prediction']),point=evaluate(root,s['point_prediction'])))+'\n')
                    manifest['snapshot_evaluations']+=2
                f.flush();g.flush();manifest['executions']+=1
            (out/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'shard{args.shard} scene{sid:02d} executions{manifest["executions"]} {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
