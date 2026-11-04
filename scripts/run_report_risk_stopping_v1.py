"""Actual train/cal trajectories, model freeze, then untouched fresh paths."""
import argparse,itertools,time
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import digest,write_json
from inspection_decisions.report_risk_episode_v1 import run_path
from inspection_decisions.report_risk_stopping_v1 import feature_row,fit_head,predict,first_stop
from inspection_decisions.costs import prefix_cost
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_component_set_planner_v1 import generate

CFG=Path('configs/report_risk_stopping_v1.json')


def sources():
    paths=[CFG,Path(__file__),Path('research/report_risk_stopping_contract_v1_2026-10-23.md'),
           Path('scripts/run_component_set_planner_v1.py'),Path('scripts/partial_surface_scene.py'),
           Path('configs/evidence_graph_dataset_v1.json'),Path('configs/comparative_evidence_interface_v1.json'),
           Path('results/component_set_planner_v1/models.json')]
    paths+=list(Path('inspection_decisions').glob('*.py'))+list(Path('comparative_evidence').glob('*.py'))
    return {str(p):digest(p) for p in paths}


def collect(cfg,kind,shard,shards):
    out=Path(cfg['output']);dest=out/f'{kind}_{shard}'
    if dest.exists():raise FileExistsError(dest)
    frozen=read(out/('frozen_models.json' if kind=='evaluation' else 'collection_contract.json'))
    for p,sha in frozen['sources'].items():assert digest(p)==sha,p
    if kind=='evaluation':assert digest(out/'models.json')==frozen['models_sha256']
    for part in ('snapshots','paths'):(dest/part).mkdir(parents=True)
    action_model=read(cfg['action_model'])['set_rank'];pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    generator=read('configs/comparative_evidence_interface_v1.json');factors=generator['factors']
    cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    start=time.perf_counter();manifest=dict(kind=kind,paths=[],backend_fits=0,scenes=[],exclusions=[],sources=sources())
    for sid in range(shard,32,shards):
        if kind=='training':
            root=Path(cfg['training_source'])/f'shard_{sid%2}'/'scenes'/f'{sid:04d}'
            split='train' if sid<16 else 'calibration';shift=False;seed=22322026+sid
        else:
            root=dest/'scenes'/f'{sid:04d}';shift=sid>=16;seed=cfg['seed_start']+sid;split='evaluation'
            generate(root,seed,cells[sid%16],generator,shift,cfg)
        manifest['scenes'].append(dict(scene=sid,root=str(root),seed=seed,split=split,shift=shift,
            hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
        for policy in ('rank','coverage'):
            saved=[]
            def snapshot(graph,state):
                target=dest/'snapshots'/f'{sid:04d}_{policy}_{state["stage"]}.json.gz'
                write_json(target,dict(graph=graph,state=state),True)
                saved.append(dict(file=str(target),sha256=digest(target)))
            result=run_path(root,policy,action_model,pairs,cfg['budget'],snapshot)
            assert result['backend_fits']==6
            # Persist the completed observed-only trajectory before truth access.
            target=dest/'paths'/f'{sid:04d}_{policy}.json';save(target,result)
            metrics=[evaluate(root,s['prediction']) for s in result['states']]
            costs=[prefix_cost(result['events'],result['metadata'],s['stage'],index_bytes=result['cost']['index_bytes']) for s in result['states']]
            labels=dest/'paths'/f'{sid:04d}_{policy}_labels.json'
            save(labels,dict(metrics=metrics,costs=costs))
            manifest['paths'].append(dict(scene=sid,policy=policy,split=split,shift=shift,file=str(target),
                sha256=digest(target),labels=str(labels),labels_sha256=digest(labels),snapshots=saved))
            manifest['backend_fits']+=6
        save(dest/'checkpoint.json',manifest)
        print(kind,'shard',shard,'scene',sid,'fits',manifest['backend_fits'],round(time.perf_counter()-start,1),'s',flush=True)
    manifest['seconds']=time.perf_counter()-start;save(dest/'manifest.json',manifest)


def load_paths(out,kind):
    result=[]
    for file in sorted(out.glob(f'{kind}_*/manifest.json')):
        manifest=read(file)
        for row in manifest['paths']:
            assert digest(row['file'])==row['sha256']
            assert digest(row['labels'])==row['labels_sha256']
            result.append(dict(**row,observed=read(row['file']),labels_data=read(row['labels'])))
    return sorted(result,key=lambda r:(r['scene'],r['policy']))


def arrays(paths,temporal):
    X=[];Y=[]
    for p in paths:
        states=p['observed']['states'];metrics=p['labels_data']['metrics'];previous=None
        for s,m in zip(states,metrics):
            X.append(feature_row(s,previous,states[0]['prediction'],temporal))
            Y.append([m['task_loss']/8,(m['task_loss']-metrics[-1]['task_loss'])/8]);previous=s['prediction']
    return np.array(X),np.array(Y)


def aggregate(metrics,stages):
    totals={k:sum(m[k] for m in metrics) for k in ('qualified_tp','fp','bad_motion_reports','task_loss')}
    totals['mean_acquisitions']=float(np.mean(stages));return totals


def select_setting(paths,method,models,settings,cfg):
    baseline=aggregate([p['labels_data']['metrics'][-1] for p in paths],[5]*len(paths));trials=[]
    for setting in settings:
        stages=[first_stop(iter(p['observed']['states']),method,models,setting,cfg['price'])[0] for p in paths]
        summary=aggregate([p['labels_data']['metrics'][s] for p,s in zip(paths,stages)],stages)
        feasible=(summary['fp']<=baseline['fp'] and summary['bad_motion_reports']<=baseline['bad_motion_reports']
                  and summary['qualified_tp']>=baseline['qualified_tp'] and summary['task_loss']<=baseline['task_loss']+1e-9)
        trials.append(dict(setting=setting,feasible=feasible,summary=summary,stages=stages))
    eligible=[r for r in trials if r['feasible']];assert eligible
    best=min(eligible,key=lambda r:(r['summary']['mean_acquisitions'],r['summary']['task_loss'],r['setting']))
    return best['setting'],dict(method=method,baseline=baseline,trials=trials,selected=best)


def train(cfg):
    out=Path(cfg['output']);assert not (out/'models.json').exists()
    contract=read(out/'collection_contract.json')
    for p,sha in contract['sources'].items():assert digest(p)==sha,p
    paths=load_paths(out,'training');assert len(paths)==64
    train=[p for p in paths if p['split']=='train'];cal=[p for p in paths if p['split']=='calibration']
    assert {p['scene'] for p in train}==set(range(16)) and {p['scene'] for p in cal}==set(range(16,32))
    models={};search=[]
    for name,temporal in [('static',False),('temporal',True)]:
        X,Y=arrays(train,temporal);C,D=arrays(cal,temporal);options=[]
        for alpha in cfg['regularization']:
            model=fit_head(X,Y,alpha);pred=np.array([predict(c,model) for c in C]);mse=float(np.mean((pred-D)**2))
            options.append((mse,-alpha,model));search.append(dict(head=name,alpha=alpha,cal_mse=mse,
                cal_risk_mse=float(np.mean((pred[:,0]-D[:,0])**2)),cal_gain_mse=float(np.mean((pred[:,1]-D[:,1])**2))))
        models[name]=min(options,key=lambda r:r[:2])[2]
    settings={};selection=[]
    for method in cfg['methods']:
        if method.startswith('full_') or method=='original_rank':settings[method]=None;continue
        policy='coverage' if method=='declared_coverage' else 'rank'
        chosen,detail=select_setting([p for p in cal if p['policy']==policy],method,models,
            list(range(6)) if method=='fixed_rank' else cfg['thresholds'],cfg)
        settings[method]=chosen;selection.append(detail)
    bundle=dict(heads=models,settings=settings)
    save(out/'models.json',bundle);save(out/'calibration_search.json',dict(heads=search,stopping=selection))
    save(out/'frozen_models.json',dict(sources=sources(),models_sha256=digest(out/'models.json'),
        search_sha256=digest(out/'calibration_search.json'),training_manifests={str(p):digest(p) for p in out.glob('training_*/manifest.json')},
        train_families=list(range(16)),calibration_families=list(range(16,32)),states_per_split=192,
        evaluation_generated=False))
    print('FROZEN',[(k,m['alpha']) for k,m in models.items()],settings,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','collect','train','evaluate'])
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();cfg=read(CFG);out=Path(cfg['output'])
    if args.stage=='init':
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True);save(out/'collection_contract.json',dict(sources=sources(),config=cfg,evaluation_generated=False))
    elif args.stage=='train':train(cfg)
    else:collect(cfg,'training' if args.stage=='collect' else 'evaluation',args.shard,args.shards)


if __name__=='__main__':main()
