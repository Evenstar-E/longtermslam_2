import csv,time
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import digest,read_graph,canonical_bytes
from inspection_decisions.component_set_v1 import representation as action_rep,decide
from inspection_decisions.component_conflict_v1 import first_stop,NEW,representation
from inspection_decisions.component_conflict_episode_v1 import run
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths
from scripts.run_component_conflict_v1 import CFG,train_models,select,component_labels

def main():
    cfg=read(CFG);out=Path(cfg['output']);assert not (out/'analysis.json').exists();checks=0
    def ck(v):
        nonlocal checks
        assert v;checks+=1
    f=read(out/'frozen_models.json');b=read(out/'models.json')
    for p,sha in f['sources'].items():ck(digest(p)==sha)
    ck(digest(out/'models.json')==f['models_sha256']);ck(digest(out/'calibration_search.json')==f['search_sha256'])
    trainpaths=load_paths(out,'training');paths=load_paths(out,'evaluation');ck(len(paths)==len(trainpaths)==64)
    tr=[p for p in trainpaths if p['split']=='train'];cal=[p for p in trainpaths if p['split']=='calibration' and p['policy']=='rank']
    ck({p['scene'] for p in tr}==set(range(16)));ck({p['scene'] for p in cal}==set(range(16,32)))
    ck(train_models(tr,cfg)==b['heads'])
    for entry in read(out/'calibration_search.json'):
        tau,detail=select(cal,entry['method'],b,cfg);ck(tau==b['settings'][entry['method']] and detail==entry)
    roots={};fits=0
    for mf in out.glob('*_*/manifest.json'):
        m=read(mf);ck(not m['exclusions']);fits+=m['backend_fits']
        for s in m['scenes']:
            root=Path(s['root']);roots[(mf.parent.name.startswith('evaluation'),s['scene'])]=root
            for rel,sha in s['hashes'].items():ck(digest(root/rel)==sha)
    pairs=read('configs/evidence_graph_dataset_v1.json')['pairs'];model=read('results/component_set_planner_v1/models.json')['set_rank']
    for p in trainpaths+paths:
        obs=p['observed'];root=roots[(p['split']=='evaluation',p['scene'])];ev=[e for e in obs['events'] if e.get('epoch')]
        previous=None;ck(len(obs['states'])==6)
        for s,snap,metric,labels in zip(obs['states'],p['snapshots'],p['labels_data']['metrics'],p['labels_data']['components']):
            ck(digest(snap['file'])==snap['sha256']);saved=read_graph(snap['file']);ck(saved['state']==s)
            ck(canonical_bytes(evaluate(root,s['prediction']))==canonical_bytes(metric));ck(component_labels(root,s['prediction'],metric)==labels)
            ck(abs(sum(labels.values())-metric['task_loss'])<1e-8)
            for method in NEW:
                ids,X=representation(s,previous,obs['states'][0]['prediction'],method);ck(np.isfinite(X).all())
            rep=action_rep(saved['graph'],pairs);_,d=decide(rep,model)
            if s['stage']<5:
                i=1+int(np.argmax(d['scores'][1:] if p['policy']=='rank' else rep['coverage'][1:]))
                ck(canonical_bytes(s['selected_option'])==canonical_bytes(rep['options'][i]));e=ev[s['stage']+2];ck([e['epoch'],e['view_id']]==list(rep['options'][i][0]))
            previous=s['prediction']
    rows=[]
    for method in cfg['methods']:
        for p in paths:
            if p['policy']!=('coverage' if method.endswith('coverage') else 'rank'):continue
            t,decisions=first_stop(iter(p['observed']['states']),method,b,b['settings'][method]);metric=p['labels_data']['metrics'][t];cost=p['labels_data']['costs'][t]
            ck(len(decisions)==t+1 and decisions[-1]['stop']);rows.append(dict(scene=p['scene'],shift=p['shift'],method=method,stage=t,metric=metric,cost=cost,decisions=decisions,
                unknown=sum(a['label']<0 for a in p['observed']['states'][t]['prediction']['assets']),objective=metric['task_loss']+.05*t))
    cand=[r for r in rows if r['method']=='component_conflict'];chosen=min([r for r in cand if r['stage']<5] or cand,key=lambda r:r['scene']);sid=chosen['scene']
    actual=run(roots[True,sid],'rank',model,pairs,cfg['budget'],lambda g,s:None,stopper=dict(method='component_conflict',models=b,setting=b['settings']['component_conflict'],price=.05))
    save(out/'physical_replay.json',actual);p=next(p for p in paths if p['scene']==sid and p['policy']=='rank')
    ck(actual['backend_fits']==chosen['stage']+1);ck(actual['states'][-1]['prediction']==p['observed']['states'][chosen['stage']]['prediction'])
    ck(abs(actual['cost']['travel_m']-chosen['cost']['total_travel_m'])<1e-9)
    summaries={};contrasts={};rng=np.random.default_rng(2313026)
    for cond in ('control','shift','all'):
        sel=[r for r in rows if cond=='all' or r['shift']==(cond=='shift')];table={}
        for method in cfg['methods']:
            g=[r for r in sel if r['method']==method];v={k:sum(r['metric'][k] for r in g) for k in ('changed','unchanged','qualified_tp','fp','bad_motion_reports','task_loss')}
            v.update(unknown=sum(r['unknown'] for r in g),mean_acquisitions=float(np.mean([r['stage'] for r in g])),mean_travel=float(np.mean([r['cost']['total_travel_m'] for r in g])),mean_objective=float(np.mean([r['objective'] for r in g])));table[method]=v
        summaries[cond]=table;n=32 if cond=='all' else 16;idx=np.c_[rng.integers(0,16,(5000,16)),rng.integers(16,32,(5000,16))] if n==32 else rng.integers(0,n,(5000,n));contrasts[cond]={}
        c=sorted([r for r in sel if r['method']=='component_conflict'],key=lambda r:r['scene'])
        for other in cfg['methods'][1:]:
            base=sorted([r for r in sel if r['method']==other],key=lambda r:r['scene']);d={}
            for name,get in {'loss':lambda r:r['metric']['task_loss'],'objective':lambda r:r['objective'],'acquisitions':lambda r:r['stage'],'travel':lambda r:r['cost']['total_travel_m']}.items():
                delta=np.array([get(x)-get(y) for x,y in zip(c,base)]);d[name]=dict(mean=float(delta.mean()),ci95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
            contrasts[cond][other]=d
    save(out/'analysis.json',dict(summaries=summaries,contrasts=contrasts,rows=rows,audit_checks=checks,training_fits=384,evaluation_fits=384,validation_fits=actual['backend_fits'],total_new_fits=fits+actual['backend_fits'],physical_scene=sid,exclusions=[]))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','method']+list(next(iter(summaries['all'].values()))))
        for cond,tab in summaries.items():
            for method,v in tab.items():writer.writerow([cond,method]+list(v.values()))
    print('AUDIT',checks,flush=True)
    for m,v in summaries['all'].items():print(m,v,flush=True)

if __name__=='__main__':main()
