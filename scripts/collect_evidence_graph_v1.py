"""Fresh split-isolated pipeline data; no learned policy/confirmation claims."""
import argparse,itertools,json,time
from pathlib import Path
from comparative_evidence.scene import generate_scene
from inspection_decisions.graph_dataset import collect_scene,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();assert 0<=args.shard<args.shards
    cfgpath=Path('configs/evidence_graph_dataset_v1.json');cfg=json.loads(cfgpath.read_text())
    generator=json.loads(Path(cfg['generator_config']).read_text());generator.update(stage='new_evidence_graph_pipeline_validation',seed_start=cfg['seed_start'])
    out=Path('results/evidence_graph_dataset_v1')/f'shard_{args.shard}'
    if out.exists():raise FileExistsError(out)
    for folder in ('scenes','inputs/states','estimator_outputs','labels','index'):(out/folder).mkdir(parents=True)
    (out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(cfg['generator_config']),Path(__file__),Path('research/evidence_graph_contract_v1_2026-10-22.md')]+list(Path('inspection_decisions').glob('*.py'))+list(Path('comparative_evidence').glob('*.py'))+[Path('scripts/partial_surface_scene.py')]
    manifest=dict(sources={str(p):digest(p) for p in paths},scenes=[],exclusions=[],states=0,transitions=0,pairs=0,
        solve_count=0,actual_payload_reads=0,shard=args.shard,shards=args.shards)
    (out/'started_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    factors=generator['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    begin=time.perf_counter()
    for sid in range(args.shard,cfg['scenes'],args.shards):
        split=next(k for k,(low,high) in cfg['splits'].items() if low<=sid<=high)
        seed=cfg['seed_start']+sid;condition=cells[sid%16];scene=out/'scenes'/f'{sid:04d}'
        generate_scene(scene,seed,condition,generator)
        records=collect_scene(scene,out,sid,split,f'family_{seed}',cfg['pairs'],cfg['budget'])
        mapping={'states':'index/states.jsonl','transitions':'inputs/transitions.jsonl',
            'state_labels':'labels/state_outcomes.jsonl','transition_labels':'labels/transition_outcomes.jsonl','pair_labels':'labels/pair_interactions.jsonl'}
        for key,relative in mapping.items():
            with (out/relative).open('a',encoding='utf-8') as f:
                for row in records[key]:f.write(json.dumps(row,allow_nan=False)+'\n')
        manifest['scenes'].append(dict(scene=sid,seed=seed,family_id=f'family_{seed}',split=split,condition=condition,
            hashes={str(p.relative_to(scene)):digest(p) for p in scene.rglob('*') if p.is_file()}))
        for key in ('states','transitions'):manifest[key]+=len(records[key])
        manifest['pairs']+=len(records['pair_labels']);manifest['solve_count']+=records['solve_count'];manifest['actual_payload_reads']+=records['actual_payload_reads']
        (out/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(f'graph shard{args.shard} scene{sid:02d} {split} states{manifest["states"]} fits{manifest["solve_count"]} {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    manifest['outputs']={str(p.relative_to(out)):digest(p) for folder in ('inputs','index','labels','estimator_outputs') for p in (out/folder).rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
