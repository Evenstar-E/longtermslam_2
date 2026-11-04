"""Frozen full policies or all initial-action forecasts; no overwrite/resume."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from comparative_evidence.policy import execute_policy
from comparative_evidence.interface import Episode, shortest_path
from comparative_evidence.selection_v8 import run_selection_v8
from comparative_evidence.geometry_modes_v2 import estimate_modes, mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_losses
from comparative_evidence.event_likelihood_v7 import held_clouds
from comparative_evidence.ray_likelihood_v8 import forecast, acquisition_scores, encode_depth, log_likelihood
from comparative_evidence.evaluation_v3 import evaluate


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def branch_scene(root, sid, cfg, out):
    ep=Episode(root,*cfg['budget']);h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
    fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
    belief=mixture_particles(fit);losses,regret=common_losses(fit,belief)
    records=held_clouds(fit,h,q,meta)
    options=[(e,v) for e in ('history','current') for v in range(1,12)]
    models={};scores={}
    for action in options:
        m=forecast(action,fit,meta,records,belief[0]);models[action]=m
        raw={}
        for shared in (True,False):
            s=acquisition_scores(m,losses,regret,shared)
            raw.update({('shared_' if shared else 'factorized_')+k:v for k,v in s.items()})
        travel=shortest_path(meta['edges'],0,action[1])[0] if action[0]=='current' else 0.
        scores[action]=dict(raw=raw,normalized={k:float(v/(1+travel/7)) for k,v in raw.items()},audit=m['audit'])
    ep.finish()
    target=out/'branches'/f'{sid:04d}_frozen_scores.json'
    # Written BEFORE any new image or cached future outcome below is read.
    target.write_text(json.dumps([dict(epoch=a[0],view=a[1],**scores[a]) for a in options],indent=2),encoding='utf-8')
    rows=[]
    for action in options:
        branch=Episode(root,*cfg['budget']);branch.act('retrieve_history',0)
        obs=branch.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1])
        model=models[action];value=encode_depth(obs,model['pixels']);branch.finish()
        nll={str(shared):float(-logsumexp(log_likelihood(model,value,shared))+np.log(len(belief[0]))) for shared in (True,False)}
        marginal=model['prob'].mean((0,1))
        brier=float(np.mean(np.sum(marginal**2,axis=1)-2*marginal[np.arange(len(value)),value]+1))
        # Same legal single action and exact old backend => use frozen v7
        # afterfit output, not rerun1056identicalfits. This is outcome reuse,
        # never a policy input. Its source hash and prediction are reevaluated.
        old=Path('results/comparative_event_branches_v7')/f'shard_{sid%2}'/'branches'/f'{sid:04d}_{action[0]}_{action[1]:02d}.json'
        saved=json.loads(old.read_text(encoding='utf-8'))
        rows.append(dict(scene=sid,epoch=action[0],view=action[1],scores=scores[action],observed_symbols=value.tolist(),
            nll=nll,depth_brier=brier,uniform_depth_brier=1-1/66,uniform_depth_nll=len(value)*np.log(66),
            cached_source=str(old),cached_sha256=sha(old),after=evaluate(root,saved['prediction']),
            initial=evaluate(root,saved['initial']),cost=branch.state()['cost'] if 'cost' in branch.state() else None,
            frozen_scores_sha256=sha(target)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['active','branches'],required=True)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();assert 0<=args.shard<args.shards
    cfgpath=Path('configs/comparative_selection_v8.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/comparative_selection_v8' if args.mode=='active' else 'results/comparative_ray_branches_v8')/f'shard_{args.shard}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/('predictions' if args.mode=='active' else 'branches')).mkdir()
    (out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(__file__)]+list(Path('comparative_evidence').glob('*.py'))
    data=[p for sid in range(args.shard,cfg['scenes'],args.shards) for p in (Path(cfg['source_data'])/f'{sid:04d}').rglob('*') if p.is_file()]
    manifest=dict(sources={str(p):sha(p) for p in paths},data={str(p):sha(p) for p in data},
                  scenes=0,executions=0,evaluations=0,branches=0,exclusions=[],mode=args.mode,shard=args.shard,shards=args.shards)
    (out/'started_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(args.shard,cfg['scenes'],args.shards):
            root=Path(cfg['source_data'])/f'{sid:04d}'
            if args.mode=='branches':
                rows=branch_scene(root,sid,cfg,out);manifest['branches']+=len(rows)
                for row in rows:f.write(json.dumps(row)+'\n')
            else:
                for method in cfg['methods']:
                    result=execute_policy(root,*cfg['budget'],lambda port:run_selection_v8(port,method))
                    p=result['prediction'];target=out/'predictions'/f'{sid:04d}_{method}.json'
                    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
                    for variant,pred in {'risk':p,**p['terminal_alternatives']}.items():
                        f.write(json.dumps(dict(scene=sid,method=method,variant=variant,**evaluate(root,pred),cost=result['cost'],
                            timings=result['timings'],prediction_sha256=sha(target)))+'\n');manifest['evaluations']+=1
                    f.flush();manifest['executions']+=1
            f.flush();manifest['scenes']+=1
            (out/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'{args.mode} shard{args.shard} scene{sid:02d} {manifest["executions"]}policies {manifest["branches"]}branches {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
