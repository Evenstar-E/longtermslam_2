"""Train/freeze, then fresh observed-data closed-loop evaluation. No overwrite."""
import argparse,itertools,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence import scene as scene_module
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import read_graph,write_json,digest
from inspection_decisions.component_set_v1 import representation,fit,decide
from inspection_decisions.component_set_episode_v1 import run_episode
from scripts.run_evidence_value_pilot_v1 import read,save,lines

CFG=Path('configs/component_set_planner_v1.json')


def sources():
    paths=[CFG,Path(__file__),Path('research/component_set_planner_contract_v1_2026-10-23.md')]
    paths+=list(Path('inspection_decisions').glob('*.py'))+list(Path('comparative_evidence').glob('*.py'))
    paths+=[Path('scripts/partial_surface_scene.py'),Path('configs/evidence_graph_dataset_v1.json'),Path('configs/comparative_evidence_interface_v1.json')]
    return {str(p):digest(p) for p in paths}


def training_groups(cfg,pairs):
    groups={s:[] for s in cfg['training_splits']};source={}
    # Existing raw outcome files mix splits. Filtering uses only index IDs;
    # old DEV outcome rows never become a training group or selection input.
    for shard in sorted(Path(cfg['training_source']).glob('shard_*')):
        records=lines(shard/'index/states.jsonl');allowed={r['state_id']:r for r in records if r['split'] in groups}
        source[str(shard/'index/states.jsonl')]=digest(shard/'index/states.jsonl')
        source[str(shard/'labels/state_outcomes.jsonl')]=digest(shard/'labels/state_outcomes.jsonl')
        outcomes={r['state_id']:r for r in lines(shard/'labels/state_outcomes.jsonl') if r['state_id'] in allowed}
        by_scene={}
        for r in allowed.values():by_scene.setdefault(r['scene'],[]).append(r)
        for sid,states in by_scene.items():
            endpoint={tuple(map(tuple,r['actions'])):r for r in states}
            prefixes=[()]+sorted({tuple(map(tuple,r['actions'][:1])) for r in states if len(r['actions'])==2})
            for prefix in prefixes:
                r=endpoint[prefix];source[r['input_file']]=digest(r['input_file'])
                assert source[r['input_file']]==r['input_sha256']
                graph=read_graph(r['input_file']);rep=representation(graph,pairs,pooled=True)
                ids=[i for i,o in enumerate(rep['options']) if prefix+o in endpoint]
                # Initial has all31; noninitial includes only recorded legal
                # second-step endpoints and STOP. Unrecorded outcomes omitted.
                L0=outcomes[r['state_id']]['metric']['task_loss']
                loss=np.array([outcomes[endpoint[prefix+rep['options'][i]]['state_id']]['metric']['task_loss'] for i in ids])
                group={key:rep[key][ids] for key in ('set','additive','pooled','coverage','counts')}
                group.update(loss=loss,gain=L0-loss,scene=sid,state_id=r['state_id'],family_id=r['family_id'])
                groups[r['split']].append(group)
    for side in groups:groups[side].sort(key=lambda r:(r['scene'],r['state_id']))
    return groups,source


def train(cfg,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    save(out/'started_training.json',dict(sources=sources(),config=cfg))
    groups,data_sources=training_groups(cfg,pairs);train,cal=[groups[k] for k in cfg['training_splits']]
    assert not {g['family_id'] for g in train}&{g['family_id'] for g in cal}
    models={};search=[]
    for kind in cfg['models']:
        best=None
        for alpha in cfg['regularization']:
            begin=time.perf_counter();model=fit(train,kind,alpha,cfg['price'])
            choices=[decide(g,model)[0] for g in cal]
            obj=float(np.mean([g['loss'][i]+cfg['price']*g['counts'][i] for g,i in zip(cal,choices)]))
            entry=dict(kind=kind,alpha=alpha,calibration_objective=obj,train_seconds=time.perf_counter()-begin,optimizer=model['optimizer'])
            search.append(entry)
            if best is None or (obj,-alpha)<best[0]:best=((obj,-alpha),model)
        models[kind]=best[1];print(kind,'alpha',best[1]['alpha'],'calobjective',best[0][0],flush=True)
    save(out/'models.json',models);save(out/'calibration_search.json',search)
    save(out/'frozen_training.json',dict(sources=sources(),data_sources=data_sources,
        models_sha256=digest(out/'models.json'),search_sha256=digest(out/'calibration_search.json'),
        groups={k:len(v) for k,v in groups.items()},rows={k:sum(len(g['gain']) for g in v) for k,v in groups.items()},
        families={k:sorted({g['family_id'] for g in v}) for k,v in groups.items()},
        evaluation_generated=False,notes='No old DEV selection. Later >one-action-prefix states are extrapolation.'))


def generate(root,seed,condition,generator,shift,cfg):
    if not shift:return scene_module.generate_scene(root,seed,condition,generator)
    original=scene_module.make_world;s=cfg['shift']
    # Predetermined seed-based layout, independent of policy results.
    rng=np.random.default_rng(seed+271);xc=float(rng.choice(s['x_centers']));w=float(rng.choice(s['widths']));zc=float(rng.choice(s['z_centers']))
    vertices=np.array([[x,y,z] for x in (xc-w/2,xc+w/2) for y in (-s['height']/2,s['height']/2) for z in (zc-s['thickness']/2,zc+s['thickness']/2)])
    def world(seed,condition):
        past,current,groups,occluders,truth=original(seed,condition)
        truth['stress_occluder_vertices']=vertices.tolist()
        return past,current,groups,[scene_module.halfspaces([vertices])],truth
    with patch.object(scene_module,'make_world',world):meta=scene_module.generate_scene(root,seed,condition,generator)
    truth=read(root/'evaluation/truth.json');target=np.array([0.,0.,5.5])
    cloud=np.concatenate([np.asarray(p) for ps in truth['current_vertices'] for p in ps]+[vertices])
    radius=float(np.max(np.linalg.norm(cloud-target,axis=1))+generator['clearance_m'])
    for poses in ([np.array(truth['poses'][f'current_{i:02d}'])[:3,3] for i in range(12)],
                  [np.array(r['nominal_T'])[:3,3] for r in meta['current']]):
        for i in range(12):
            a,b=poses[i],poses[(i+1)%12];d=b-a;t=np.clip(np.dot(target-a,d)/np.dot(d,d),0,1)
            assert np.linalg.norm(a+t*d-target)>radius,'Shifted occluder path not clear'
    truth['path_enclosing_radius_m']=radius;save(root/'evaluation/truth.json',truth)
    return meta


def run(cfg,out,shard,shards):
    freeze=read(out/'frozen_training.json')
    for p,sha in freeze['sources'].items():assert digest(p)==sha,p
    assert digest(out/'models.json')==freeze['models_sha256']
    dest=out/f'shard_{shard}'
    if dest.exists():raise FileExistsError(dest)
    for part in ('scenes','episodes','snapshots'):(dest/part).mkdir(parents=True)
    models=read(out/'models.json');pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    generator=read('configs/comparative_evidence_interface_v1.json');factors=generator['factors']
    cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    manifest=dict(sources=freeze['sources'],models_sha256=freeze['models_sha256'],scenes=[],episodes=0,backend_fits=0,exclusions=[])
    begin=time.perf_counter();save(dest/'started_manifest.json',manifest)
    with (dest/'metrics.jsonl').open('w',encoding='utf-8') as stream:
        for sid in range(shard,cfg['new_scenes'],shards):
            root=dest/'scenes'/f'{sid:04d}';seed=cfg['seed_start']+sid;shift=sid>=16
            generate(root,seed,cells[sid%16],generator,shift,cfg)
            manifest['scenes'].append(dict(scene=sid,seed=seed,shift=shift,condition=cells[sid%16],hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
            for method in cfg['methods']:
                def snapshot(graph,prediction,record):
                    stem=f'{sid:04d}_{method}_{record["stage"]}'
                    write_json(dest/'snapshots'/f'{stem}.json.gz',dict(graph=graph,prediction=prediction,decision=record),True)
                result=run_episode(root,method,models,pairs,cfg['budget'],snapshot)
                # Entire actor finished and persisted before evaluation truth.
                target=dest/'episodes'/f'{sid:04d}_{method}.json';save(target,result)
                metric=evaluate(root,result['prediction'])
                row=dict(scene=sid,seed=seed,shift=shift,method=method,metric=metric,cost=result['cost'],timings=result['timings'],
                         fits=result['backend_fits'],episode_sha256=digest(target),stages=len(result['trajectory']),
                         extrapolation=[r['details'].get('out_of_train_range_fraction') for r in result['trajectory']])
                stream.write(json.dumps(row)+'\n');stream.flush();manifest['episodes']+=1;manifest['backend_fits']+=result['backend_fits']
            save(dest/'checkpoint.json',manifest)
            print(f'set shard{shard} scene{sid} shift{shift} episodes{manifest["episodes"]} fits{manifest["backend_fits"]} {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    manifest['outputs']={str(p.relative_to(dest)):digest(p) for directory in ('episodes','snapshots') for p in (dest/directory).glob('*')}
    save(dest/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','run']);parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();cfg=read(CFG);out=Path(cfg['output'])
    if args.stage=='train':train(cfg,out)
    else:
        assert 0<=args.shard<args.shards
        run(cfg,out,args.shard,args.shards)


if __name__=='__main__':main()
