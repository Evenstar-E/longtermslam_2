"""Frozen all-method evaluation, exact target replay, actual early-stop check."""
import csv,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import digest,read_graph,canonical_bytes
from inspection_decisions.component_set_v1 import representation,decide
from inspection_decisions.costs import prefix_cost
from inspection_decisions.optional_stopping_v1 import first_stop,decision,train_banks
from inspection_decisions.report_risk_episode_v1 import run_path
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths
from scripts.run_optional_stopping_v1 import CFG,select_setting


def main():
    cfg=read(CFG);out=Path(cfg['output'])
    if (out/'analysis.json').exists():raise FileExistsError(out/'analysis.json')
    start=time.perf_counter();checks=0
    def check(ok,message):
        nonlocal checks
        if not ok:raise AssertionError(message)
        checks+=1
    frozen=read(out/'frozen_models.json');bundle=read(out/'models.json')
    for p,sha in frozen['sources'].items():check(digest(p)==sha,p)
    for name,key in [('models.json','models_sha256'),('calibration_search.json','search_sha256'),('training_targets.json','targets_sha256')]:
        check(digest(out/name)==frozen[key],name)
    for p,sha in frozen['training_manifests'].items():check(digest(p)==sha,p)
    training=load_paths(Path(cfg['training_paths']),'training');paths=load_paths(out,'evaluation')
    train=[p for p in training if p['split']=='train'];cal=[p for p in training if p['split']=='calibration' and p['policy']=='rank']
    check({p['scene'] for p in train}==set(range(16)) and {p['scene'] for p in cal}==set(range(16,32)),'family split')
    heads,target_audit=train_banks(train,bundle['legacy'],cfg['alpha'],cfg['price'],cfg['folds'])
    check(heads==bundle['heads'],'exact train replay')
    saved_targets=read(out/'training_targets.json');check(target_audit==saved_targets,'exact crossfit target replay')
    for name in ('optional_cf_temporal','optional_cf_static'):
        detail=saved_targets[name];rows=detail['rows'];targets=np.array(detail['targets'])
        for teacher in detail['teachers']:
            tr=set(teacher['training_families']);held=set(teacher['held_families'])
            check(len(tr)==12 and len(held)==4 and not tr&held,'excluded teacher families')
            check(tr|held==set(range(16)),'complete fold')
            check(all(f%4==teacher['fold'] for f in held),'fold definition')
        check(len(rows)==32 and targets.shape==(32,5),'targets size')
        for row in rows:check(row['family'] in range(16),'no cal or test target')
    for row in read(out/'calibration_search.json'):
        setting,detail=select_setting(cal,row['method'],bundle,cfg)
        check(setting==bundle['settings'][row['method']] and detail==row,'threshold selection replay')
    check(bundle['legacy']==read(cfg['legacy_models']),'frozen prior baselines')
    check(len(paths)==64,'64 fresh full paths')
    check({(p['scene'],p['policy']) for p in paths}=={(i,k) for i in range(32) for k in ('rank','coverage')},'no missing family/path')
    scene_roots={};seeds=[];newfits=0
    for mf in sorted(out.glob('evaluation_*/manifest.json')):
        m=read(mf);check(m['exclusions']==[],'no exclusions');newfits+=m['backend_fits']
        for s in m['scenes']:
            root=Path(s['root']);scene_roots[s['scene']]=root;seeds.append(s['seed'])
            check(s['seed']==cfg['seed_start']+s['scene'],'new seed')
            check(s['shift']==(s['scene']>=16),'condition')
            for rel,sha in s['hashes'].items():check(digest(root/rel)==sha,'raw scene provenance')
    check(set(seeds)==set(range(cfg['seed_start'],cfg['seed_start']+32)) and len(seeds)==32,'unique prospective seeds')
    check(newfits==384,'actual new fit count')
    pairs=read('configs/evidence_graph_dataset_v1.json')['pairs'];action_model=read(cfg['action_model'])['set_rank']
    for p in paths:
        obs=p['observed'];root=scene_roots[p['scene']];events=[e for e in obs['events'] if e.get('epoch')]
        check(len(obs['states'])==len(p['snapshots'])==len(p['labels_data']['metrics'])==6,'complete6states')
        check(len(events)==7,'seven actual payloads')
        for e in events:
            token=obs['metadata'][e['epoch']][e['view_id']]['token']
            check(digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'],'actual payload')
        for state,snap,metric in zip(obs['states'],p['snapshots'],p['labels_data']['metrics']):
            t=state['stage'];check(digest(snap['file'])==snap['sha256'],'snapshot hash')
            o=read_graph(snap['file']);check(canonical_bytes(o['state'])==canonical_bytes(state),'pre-action persistence')
            rep=representation(o['graph'],pairs);_,detail=decide(rep,action_model)
            if t<5:
                rank=1+int(np.argmax(detail['scores'][1:]));cov=1+int(np.argmax(rep['coverage'][1:]))
                choice=rank if p['policy']=='rank' else cov
                check(abs(state['best_rank_score']-detail['scores'][rank])<1e-12,'rank score')
                check(abs(state['best_coverage_score']-rep['coverage'][cov])<1e-12,'coverage score')
                check(canonical_bytes(state['selected_option'])==canonical_bytes(rep['options'][choice]),'frozen selection')
                e=events[t+2];check([e['epoch'],e['view_id']]==list(rep['options'][choice][0]),'executed first action')
            else:check(not state['selected_option'] and len(rep['options'])==1,'budget STOP')
            check(canonical_bytes(evaluate(root,state['prediction']))==canonical_bytes(metric),'all endpoints reevaluated')
            cost=prefix_cost(obs['events'],obs['metadata'],t,index_bytes=obs['cost']['index_bytes'])
            check(cost==p['labels_data']['costs'][t],'prefix return cost')
        check(abs(p['labels_data']['costs'][5]['total_travel_m']-obs['cost']['travel_m'])<1e-10,'full actual return')
    rows=[]
    for method in cfg['methods']:
        policy='coverage' if method.endswith('coverage') else 'rank'
        for p in paths:
            if p['policy']!=policy:continue
            stage,decisions=first_stop(iter(p['observed']['states']),method,bundle,bundle['settings'][method],cfg['price'])
            check(len(decisions)==stage+1 and decisions[-1]['stop'],'causal first stop')
            metric=p['labels_data']['metrics'][stage];cost=p['labels_data']['costs'][stage]
            check(cost['history_views']-1+cost['additional_current_views']==stage,'prefix count')
            rows.append(dict(scene=p['scene'],shift=p['shift'],method=method,stage=stage,metric=metric,cost=cost,
                unknown=sum(a['label']<0 for a in p['observed']['states'][stage]['prediction']['assets']),
                objective=metric['task_loss']+cfg['price']*stage,decisions=decisions))
    # Old full-path runner calls this adapter only for actual physical STOP replay.
    def actual_stop(method,state,previous,initial,models,setting,price):
        return decision(method,state,previous,initial,models,setting)
    p=next(p for p in paths if p['scene']==0 and p['policy']=='rank')
    candidate=next(r for r in rows if r['scene']==0 and r['method']=='optional_cf_temporal')
    with patch('inspection_decisions.report_risk_episode_v1.should_stop',actual_stop):
        replay=run_path(scene_roots[0],'rank',action_model,pairs,cfg['budget'],lambda g,s:None,
            stopper=dict(method='optional_cf_temporal',models=bundle,setting=bundle['settings']['optional_cf_temporal'],price=cfg['price']))
    save(out/'physical_replay_scene0.json',replay)
    check(replay['backend_fits']==candidate['stage']+1,'physical stop stage')
    check(canonical_bytes(replay['states'][-1]['prediction'])==canonical_bytes(p['observed']['states'][candidate['stage']]['prediction']),'physical report')
    check(abs(replay['cost']['travel_m']-candidate['cost']['total_travel_m'])<1e-10,'physical return')
    check(replay['cost']['payload_bytes']==candidate['cost']['payload_bytes'],'physical bytes')
    check(canonical_bytes(evaluate(scene_roots[0],replay['states'][-1]['prediction']))==canonical_bytes(candidate['metric']),'physical endpoint labels')
    summaries={};contrasts={};rng=np.random.default_rng(cfg['bootstrap']['seed']);B=cfg['bootstrap']['replicates']
    for condition in ('control','shift','all'):
        selected=[r for r in rows if condition=='all' or r['shift']==(condition=='shift')];table={}
        for method in cfg['methods']:
            group=[r for r in selected if r['method']==method]
            v={k:sum(r['metric'][k] for r in group) for k in ('changed','unchanged','qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references')}
            v.update(families=len(group),unknown=sum(r['unknown'] for r in group),
                mean_acquisitions=float(np.mean([r['stage'] for r in group])),
                mean_return_travel_m=float(np.mean([r['cost']['total_travel_m'] for r in group])),
                mean_objective=float(np.mean([r['objective'] for r in group])))
            table[method]=v
        summaries[condition]=table;n=table['optional_cf_temporal']['families']
        idx=np.c_[rng.integers(0,16,(B,16)),rng.integers(16,32,(B,16))] if condition=='all' else rng.integers(0,n,(B,n))
        contrasts[condition]={}
        cand=sorted([r for r in selected if r['method']=='optional_cf_temporal'],key=lambda r:r['scene'])
        getters={'task_loss':lambda r:r['metric']['task_loss'],'objective':lambda r:r['objective'],
            'acquisitions':lambda r:r['stage'],'travel_m':lambda r:r['cost']['total_travel_m'],
            'qualified_tp':lambda r:r['metric']['qualified_tp'],'fp':lambda r:r['metric']['fp'],
            'bad_motion':lambda r:r['metric']['bad_motion_reports']}
        for baseline in cfg['methods'][1:]:
            base=sorted([r for r in selected if r['method']==baseline],key=lambda r:r['scene']);d={}
            for name,get in getters.items():
                delta=np.array([get(a)-get(b) for a,b in zip(cand,base)])
                d[name]=dict(mean=float(delta.mean()),ci95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),
                    positive=int(sum(delta>1e-10)),negative=int(sum(delta< -1e-10)),zero=int(sum(abs(delta)<=1e-10)))
            contrasts[condition][baseline]=d
    result=dict(summaries=summaries,contrasts=contrasts,rows=rows,audit_checks=checks,
        current_training_fits=0,reused_training_calibration_fits=384,actual_evaluation_fits=newfits,
        actual_validation_fits=replay['backend_fits'],fresh_families=32,shared_full_paths=64,
        stopping_policy_outcomes=len(rows),exclusions=[],seconds=time.perf_counter()-start,
        bootstrap='5000 paired family, all32 condition-stratified, exploratory unadjusted95%')
    save(out/'analysis.json',result)
    fields=['condition','method','families','changed','unchanged','qualified_tp','fp','bad_motion_reports','unknown','task_loss','mean_acquisitions','mean_return_travel_m','mean_objective']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for condition,table in summaries.items():
            for method,v in table.items():writer.writerow(dict(condition=condition,method=method,**v))
    print('AUDIT',checks,'actual new fits',newfits+replay['backend_fits'],flush=True)
    for method,v in summaries['all'].items():print(method,v,flush=True)


if __name__=='__main__':main()
