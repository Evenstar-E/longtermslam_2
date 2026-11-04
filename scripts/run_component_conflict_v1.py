import argparse,itertools,time
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import digest,write_json,canonical_bytes
from inspection_decisions.component_conflict_episode_v1 import run
from inspection_decisions.component_conflict_v1 import NEW,representation,fit,first_stop
from inspection_decisions.costs import prefix_cost
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_component_set_planner_v1 import generate
from scripts.run_report_risk_stopping_v1 import load_paths,aggregate
CFG='configs/component_conflict_v1.json'

def sources():
    paths=[Path(CFG),Path(__file__),Path('research/component_conflict_contract_v1_2026-10-23.md'),Path('results/optional_stopping_v1/models.json'),Path('results/component_set_planner_v1/models.json')]
    paths+=list(Path('inspection_decisions').glob('*.py'))+list(Path('comparative_evidence').glob('*.py'))
    return {str(p):digest(p) for p in paths}

def component_labels(root,prediction,metric):
    truth=read(root/'evaluation/truth.json');by={a['ref_id']:a for a in prediction['assets']};loss={}
    for r in metric['ownership_records']:
        j=r['owner'];label=by[r['ref_id']]['label']
        if j<0 or not r['identity_ok']:v=.35 if label<0 else 1.
        elif label<0:v=.35
        elif label==0:v=float(truth['changed'][j])
        else:v=.8 if not truth['changed'][j] else float(r['motion_error_m']>.10)
        loss[str(r['ref_id'])]=v
    residual=metric['task_loss']-sum(loss.values());assert residual>=-1e-8
    loss={j:v+max(residual,0)/len(loss) for j,v in loss.items()}
    assert abs(sum(loss.values())-metric['task_loss'])<1e-8
    return loss

def collect(cfg,kind,shard):
    out=Path(cfg['output']);f=read(out/('frozen_models.json' if kind=='evaluation' else 'collection_contract.json'))
    for p,sha in f['sources'].items():assert digest(p)==sha,p
    dest=out/f'{kind}_{shard}';assert not dest.exists()
    for folder in ('paths','snapshots'):(dest/folder).mkdir(parents=True)
    model=read('results/component_set_planner_v1/models.json')['set_rank'];pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    generator=read('configs/comparative_evidence_interface_v1.json');factors=generator['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    m=dict(paths=[],scenes=[],backend_fits=0,exclusions=[]);begin=time.perf_counter()
    for sid in range(shard,32,2):
        root=Path('results/evidence_graph_dataset_v1')/f'shard_{sid%2}'/'scenes'/f'{sid:04d}' if kind=='training' else dest/'scenes'/f'{sid:04d}'
        shift=kind=='evaluation' and sid>=16;seed=(22322026 if kind=='training' else cfg['seed_start'])+sid
        if kind=='evaluation':generate(root,seed,cells[sid%16],generator,shift,cfg)
        split=('train' if sid<16 else 'calibration') if kind=='training' else 'evaluation'
        m['scenes'].append(dict(scene=sid,root=str(root),seed=seed,shift=shift,hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
        for policy in ('rank','coverage'):
            snapshots=[]
            def snap(graph,state):
                target=dest/'snapshots'/f'{sid:04d}_{policy}_{state["stage"]}.json.gz';write_json(target,dict(graph=graph,state=state),True)
                snapshots.append(dict(file=str(target),sha256=digest(target)))
            obs=run(root,policy,model,pairs,cfg['budget'],snap);target=dest/'paths'/f'{sid:04d}_{policy}.json';save(target,obs)
            metrics=[evaluate(root,s['prediction']) for s in obs['states']]
            labels=dict(metrics=metrics,components=[component_labels(root,s['prediction'],v) for s,v in zip(obs['states'],metrics)],
                costs=[prefix_cost(obs['events'],obs['metadata'],s['stage'],index_bytes=obs['cost']['index_bytes']) for s in obs['states']])
            lf=dest/'paths'/f'{sid:04d}_{policy}_labels.json';save(lf,labels)
            if kind=='training':
                old=read(Path('results/report_risk_stopping_v1')/f'training_{sid%2}'/'paths'/f'{sid:04d}_{policy}.json')
                assert all(canonical_bytes(s['prediction'])==canonical_bytes(t['prediction']) for s,t in zip(obs['states'],old['states']))
            m['paths'].append(dict(scene=sid,policy=policy,split=split,shift=shift,file=str(target),sha256=digest(target),labels=str(lf),labels_sha256=digest(lf),snapshots=snapshots));m['backend_fits']+=obs['backend_fits']
        save(dest/'checkpoint.json',m);print(kind,shard,sid,m['backend_fits'],round(time.perf_counter()-begin,1),flush=True)
    save(dest/'manifest.json',m)

def train_models(paths,cfg):
    heads={}
    for method in NEW:
        groups=[([],[],[]) for _ in range(6)]
        for p in paths:
            states=p['observed']['states'];labels=p['labels_data'];previous=None
            for t,s in enumerate(states):
                ids,X=representation(s,previous,states[0]['prediction'],method)
                y=np.array([labels['metrics'][t]['task_loss']/8]) if method=='global_augmented' else np.array([labels['components'][t][str(j)] for j in ids])
                groups[0][0].extend(X);groups[0][1].extend(y);groups[0][2].extend([1/len(X)]*len(X))
                if t<5:
                    ny=[labels['metrics'][t+1]['task_loss']/8] if method=='global_augmented' else [labels['components'][t+1][str(j)] for j in ids]
                    groups[t+1][0].extend(X);groups[t+1][1].extend(ny);groups[t+1][2].extend([1/len(X)]*len(X))
                previous=s['prediction']
        heads[method]=dict(risk=fit(*groups[0],cfg['alpha']),next=[fit(*g,cfg['alpha']) for g in groups[1:]])
    return heads

def select(cal,method,bundle,cfg):
    base=aggregate([p['labels_data']['metrics'][-1] for p in cal],[5]*len(cal));trials=[]
    for tau in cfg['thresholds']:
        stages=[first_stop(iter(p['observed']['states']),method,bundle,tau)[0] for p in cal]
        s=aggregate([p['labels_data']['metrics'][t] for p,t in zip(cal,stages)],stages)
        ok=s['fp']<=base['fp'] and s['bad_motion_reports']<=base['bad_motion_reports'] and s['qualified_tp']>=base['qualified_tp'] and s['task_loss']<=base['task_loss']+1e-9
        trials.append(dict(tau=tau,summary=s,feasible=ok,stages=stages))
    best=min([t for t in trials if t['feasible']],key=lambda t:(t['summary']['mean_acquisitions'],t['summary']['task_loss'],t['tau']))
    return best['tau'],dict(method=method,trials=trials,selected=best)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','collect','train','evaluate']);ap.add_argument('--shard',type=int,default=0);args=ap.parse_args();cfg=read(CFG);out=Path(cfg['output'])
    if args.stage=='init':
        assert not out.exists();out.mkdir();save(out/'collection_contract.json',dict(sources=sources()))
    elif args.stage=='train':
        assert not (out/'models.json').exists();paths=load_paths(out,'training');assert len(paths)==64
        train=[p for p in paths if p['split']=='train'];cal=[p for p in paths if p['split']=='calibration' and p['policy']=='rank']
        assert {p['scene'] for p in train}==set(range(16)) and {p['scene'] for p in cal}==set(range(16,32))
        legacy=read('results/optional_stopping_v1/models.json');b=dict(heads=train_models(train,cfg),legacy=legacy,settings=dict(legacy['settings']));search=[]
        for method in NEW:
            tau,detail=select(cal,method,b,cfg);b['settings'][method]=tau;search.append(detail);print(method,tau,detail['selected']['summary'],flush=True)
        save(out/'models.json',b);save(out/'calibration_search.json',search);save(out/'frozen_models.json',dict(sources=sources(),models_sha256=digest(out/'models.json'),search_sha256=digest(out/'calibration_search.json'),training_manifests={str(p):digest(p) for p in out.glob('training_*/manifest.json')}))
    else:collect(cfg,'training' if args.stage=='collect' else 'evaluation',args.shard)

if __name__=='__main__':main()
