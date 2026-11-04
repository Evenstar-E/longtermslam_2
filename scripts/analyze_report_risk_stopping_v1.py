"""Frozen causal-prefix evaluation, source audit, paired family uncertainty."""
import csv,time
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import digest,read_graph,canonical_bytes
from inspection_decisions.component_set_v1 import representation,decide
from inspection_decisions.report_risk_stopping_v1 import first_stop,predict,fit_head
from inspection_decisions.report_risk_episode_v1 import run_path
from inspection_decisions.costs import prefix_cost
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import CFG,load_paths,arrays,select_setting


def main():
    cfg=read(CFG);out=Path(cfg['output']);target=out/'analysis.json'
    if target.exists():raise FileExistsError(target)
    begin=time.perf_counter();checks=0
    def check(ok,message):
        nonlocal checks
        if not ok:raise AssertionError(message)
        checks+=1
    frozen=read(out/'frozen_models.json');bundle=read(out/'models.json');models=bundle['heads'];settings=bundle['settings']
    for p,sha in frozen['sources'].items():check(digest(p)==sha,p)
    check(digest(out/'models.json')==frozen['models_sha256'],'models freeze')
    check(digest(out/'calibration_search.json')==frozen['search_sha256'],'search freeze')
    for p,sha in frozen['training_manifests'].items():check(digest(p)==sha,p)
    paths=load_paths(out,'evaluation');training=load_paths(out,'training')
    check(len(paths)==64 and len(training)==64,'complete paths')
    check({(p['scene'],p['policy']) for p in paths}=={(s,k) for s in range(32) for k in ('rank','coverage')},'all families and policies')
    train=[p for p in training if p['split']=='train'];cal=[p for p in training if p['split']=='calibration']
    check({p['scene'] for p in train}==set(range(16)) and {p['scene'] for p in cal}==set(range(16,32)),'split isolation')
    for name in ('static','temporal'):
        X,Y=arrays(train,name=='temporal');model=fit_head(X,Y,models[name]['alpha'])
        check(model==models[name],name+' exact retrain')
    search=read(out/'calibration_search.json');recomputed=[]
    for name in ('static','temporal'):
        X,Y=arrays(train,name=='temporal');C,D=arrays(cal,name=='temporal');candidates=[]
        for alpha in cfg['regularization']:
            m=fit_head(X,Y,alpha);P=np.array([predict(c,m) for c in C]);mse=float(np.mean((P-D)**2))
            entry=next(r for r in search['heads'] if r['head']==name and r['alpha']==alpha)
            check(abs(mse-entry['cal_mse'])<1e-14,'alpha MSE replay');candidates.append((mse,-alpha))
        check(-min(candidates)[1]==models[name]['alpha'],'alpha selection')
    for method in cfg['methods']:
        if method.startswith('full_') or method=='original_rank':continue
        policy='coverage' if method=='declared_coverage' else 'rank'
        selected,detail=select_setting([p for p in cal if p['policy']==policy],method,models,
            list(range(6)) if method=='fixed_rank' else cfg['thresholds'],cfg)
        check(selected==settings[method],'threshold replay '+method)
        check(detail==next(r for r in search['stopping'] if r['method']==method),'calibration search replay')
    scene_roots={};all_seeds={}
    for kind in ('training','evaluation'):
        seeds=[]
        for mf in sorted(out.glob(f'{kind}_*/manifest.json')):
            manifest=read(mf);check(manifest['exclusions']==[],'no exclusions')
            for scene in manifest['scenes']:
                root=Path(scene['root']);seeds.append(scene['seed'])
                if kind=='evaluation':scene_roots[scene['scene']]=root
                for rel,sha in scene['hashes'].items():check(digest(root/rel)==sha,'scene integrity '+rel)
        check(len(seeds)==len(set(seeds))==32,'unique seeds');all_seeds[kind]=seeds
    check(not set(all_seeds['training'])&set(all_seeds['evaluation']),'family seed independence')
    check(set(all_seeds['evaluation'])==set(range(cfg['seed_start'],cfg['seed_start']+32)),'prospective seeds')
    pairs=read('configs/evidence_graph_dataset_v1.json')['pairs'];action_model=read(cfg['action_model'])['set_rank']
    for p in training+paths:
        observed=p['observed']
        root=scene_roots[p['scene']] if p['split']=='evaluation' else Path(cfg['training_source'])/f'shard_{p["scene"]%2}'/'scenes'/f'{p["scene"]:04d}'
        check(len(observed['states'])==6,'six actual states')
        check(len(p['snapshots'])==len(p['labels_data']['metrics'])==6,'complete snapshot/label endpoints')
        events=[e for e in observed['events'] if e.get('epoch')]
        check(len(events)==7,'preview H0 plus five actions')
        for event in events:
            token=observed['metadata'][event['epoch']][event['view_id']]['token']
            check(digest(root/'payloads'/f'{token}.npz')==event['payload_sha256'],'actual payload provenance')
        for state,snap,metric in zip(observed['states'],p['snapshots'],p['labels_data']['metrics']):
            check(digest(snap['file'])==snap['sha256'],'observed snapshot integrity')
            obj=read_graph(snap['file']);check(canonical_bytes(obj['state'])==canonical_bytes(state),'saved-before-action state')
            rep=representation(obj['graph'],pairs);_,detail=decide(rep,action_model)
            if len(rep['options'])>1:
                rank=1+int(np.argmax(detail['scores'][1:]));cov=1+int(np.argmax(rep['coverage'][1:]))
                choice=rank if p['policy']=='rank' else cov
                check(abs(state['best_rank_score']-detail['scores'][rank])<1e-12,'rank score replay')
                check(abs(state['best_coverage_score']-rep['coverage'][cov])<1e-12,'coverage score replay')
                check(canonical_bytes(state['selected_option'])==canonical_bytes(rep['options'][choice]),'frozen acquisition replay')
                event=events[state['stage']+2]
                check([event['epoch'],event['view_id']]==list(rep['options'][choice][0]),'actually executed first action')
            else:check(state['stage']==5 and not state['selected_option'],'terminal budget')
            check(canonical_bytes(evaluate(root,state['prediction']))==canonical_bytes(metric),'all endpoint labels reevaluated')
            cost=prefix_cost(observed['events'],observed['metadata'],state['stage'],index_bytes=observed['cost']['index_bytes'])
            check(cost==p['labels_data']['costs'][state['stage']],'return-aware prefix accounting')
        c=p['labels_data']['costs'][-1]
        check(abs(c['total_travel_m']-observed['cost']['travel_m'])<1e-10,'full actual vs prefix return cost')
        check(c['payload_bytes']==observed['cost']['payload_bytes'],'full actual bytes')
    rows=[]
    for method in cfg['methods']:
        policy='coverage' if method.endswith('coverage') else 'rank'
        for p in paths:
            if p['policy']!=policy:continue
            stage,decisions=first_stop(iter(p['observed']['states']),method,models,settings[method],cfg['price'])
            check(len(decisions)==stage+1 and decisions[-1]['stop'],'causal first stop')
            metric=p['labels_data']['metrics'][stage];cost=p['labels_data']['costs'][stage]
            pred=p['observed']['states'][stage]['prediction']
            rows.append(dict(method=method,scene=p['scene'],shift=p['shift'],stage=stage,metric=metric,cost=cost,
                unknown=sum(a['label']<0 for a in pred['assets']),objective=metric['task_loss']+cfg['price']*stage,decisions=decisions))
    # Physical candidate rerun is independent of shared-path replay, no truth input.
    p=next(p for p in paths if p['scene']==0 and p['policy']=='rank')
    row=next(r for r in rows if r['scene']==0 and r['method']=='dual_temporal')
    actual=run_path(scene_roots[0],'rank',action_model,pairs,cfg['budget'],lambda g,s:None,
        stopper=dict(method='dual_temporal',models=models,setting=settings['dual_temporal'],price=cfg['price']))
    save(out/'physical_replay_scene0.json',actual)
    check(len(actual['states'])==row['stage']+1,'actual first-stop stage')
    check(canonical_bytes(actual['states'][-1]['prediction'])==canonical_bytes(p['observed']['states'][row['stage']]['prediction']),'actual endpoint prediction')
    check(abs(actual['cost']['travel_m']-row['cost']['total_travel_m'])<1e-10,'actual endpoint return')
    check(actual['cost']['payload_bytes']==row['cost']['payload_bytes'],'actual endpoint bytes')
    check(canonical_bytes(evaluate(scene_roots[0],actual['states'][-1]['prediction']))==canonical_bytes(row['metric']),'actual endpoint metrics')
    summaries={};contrasts={};rng=np.random.default_rng(cfg['bootstrap']['seed']);B=cfg['bootstrap']['replicates']
    for condition in ('control','shift','all'):
        selected=[r for r in rows if condition=='all' or r['shift']==(condition=='shift')]
        table={}
        for method in cfg['methods']:
            group=[r for r in selected if r['method']==method];n=len(group)
            v={k:sum(r['metric'][k] for r in group) for k in ('changed','unchanged','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references')}
            v.update(families=n,unknown=sum(r['unknown'] for r in group),mean_acquisitions=float(np.mean([r['stage'] for r in group])),
                mean_return_travel_m=float(np.mean([r['cost']['total_travel_m'] for r in group])),mean_objective=float(np.mean([r['objective'] for r in group])))
            table[method]=v
        summaries[condition]=table
        n=table['dual_temporal']['families']
        indices=np.c_[rng.integers(0,16,(B,16)),rng.integers(16,32,(B,16))] if condition=='all' else rng.integers(0,n,(B,n))
        contrasts[condition]={}
        candidate=sorted([r for r in selected if r['method']=='dual_temporal'],key=lambda r:r['scene'])
        for baseline in [m for m in cfg['methods'] if m not in ('dual_temporal','declared_coverage')]:
            base=sorted([r for r in selected if r['method']==baseline],key=lambda r:r['scene']);d={}
            getters={'task_loss':lambda r:r['metric']['task_loss'],'objective':lambda r:r['objective'],
                'acquisitions':lambda r:r['stage'],'travel_m':lambda r:r['cost']['total_travel_m'],
                'qualified_tp':lambda r:r['metric']['qualified_tp'],'fp':lambda r:r['metric']['fp'],
                'bad_motion':lambda r:r['metric']['bad_motion_reports']}
            for name,get in getters.items():
                delta=np.array([get(a)-get(b) for a,b in zip(candidate,base)])
                d[name]=dict(mean=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
                             positive=int(sum(delta>1e-10)),negative=int(sum(delta< -1e-10)),zero=int(sum(abs(delta)<=1e-10)))
            contrasts[condition][baseline]=d
    forecast={}
    for name in ('static','temporal'):
        X,Y=arrays(paths,name=='temporal');P=np.array([predict(x,models[name]) for x in X])
        forecast[name]=dict(risk_mse=float(np.mean((P[:,0]-Y[:,0])**2)),gain_mse=float(np.mean((P[:,1]-Y[:,1])**2)),
            note='All fresh states on both full paths; dependent within family, diagnostic only')
    result=dict(summaries=summaries,contrasts=contrasts,rows=rows,forecast=forecast,audit_checks=checks,
        actual_training_fits=384,actual_evaluation_fits=384,actual_validation_fits=actual['backend_fits'],
        fresh_families=32,shared_full_paths=64,stopping_policy_outcomes=len(rows),exclusions=[],
        bootstrap='5000 paired family, all stratified by condition, exploratory unadjusted95%; not independent states',seconds=time.perf_counter()-begin)
    save(target,result)
    fields=['condition','method','families','changed','unchanged','qualified_tp','fp','bad_motion_reports','unknown','task_loss','mean_acquisitions','mean_return_travel_m','mean_objective']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for condition,table in summaries.items():
            for method,v in table.items():writer.writerow(dict(condition=condition,method=method,**v))
    print('AUDIT',checks,'actual validation fits',actual['backend_fits'],flush=True)
    for method,v in summaries['all'].items():print(method,v,flush=True)


if __name__=='__main__':main()
