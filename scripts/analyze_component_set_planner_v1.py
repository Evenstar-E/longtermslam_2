"""Full trace replay/audit and paired, exploratory closed-loop comparisons."""
from pathlib import Path
from unittest.mock import patch
import csv,json
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.graph_dataset import read_graph,digest,canonical_bytes
from inspection_decisions.component_set_v1 import representation,decide,fit
from inspection_decisions.component_set_episode_v1 import run_episode
from scripts.run_component_set_planner_v1 import training_groups
from scripts.run_evidence_value_pilot_v1 import read,save,lines


def summarize(rows,price):
    result={k:sum(r['metric'][k] for r in rows) for k in ('changed','unchanged','qualified_tp','fp','bad_motion_reports','known','identity_wrong_references','task_loss')}
    result['families']=len(rows);result['unknown']=8*len(rows)-result['known']
    result['mean_acquisitions']=float(np.mean([r['cost']['history_views']-1+r['cost']['additional_current_views'] for r in rows]))
    result['mean_objective']=result['task_loss']/len(rows)+price*result['mean_acquisitions']
    for key in ('travel_m','payload_bytes'):result['mean_'+key]=float(np.mean([r['cost'][key] for r in rows]))
    for key in ('backend_s','actor_including_graph_s'):result['mean_'+key]=float(np.mean([r['timings'][key] for r in rows]))
    result['backend_fits']=sum(r['fits'] for r in rows)
    result['beyond_training_prefix_states']=sum(max(0,r['stages']-2) for r in rows)
    result['total_decision_states']=sum(r['stages'] for r in rows)
    values=[v for r in rows for v in r['extrapolation'] if v is not None]
    result['mean_out_of_train_range_feature_fraction']=float(np.mean(values)) if values else None
    return result


def main():
    cfg=read('configs/component_set_planner_v1.json');out=Path(cfg['output']);frozen=read(out/'frozen_training.json')
    if (out/'analysis.json').exists():raise FileExistsError('Analysis exists; do not replace evaluated results')
    checks=0
    for p,sha in {**frozen['sources'],**frozen['data_sources']}.items():assert digest(p)==sha,p;checks+=1
    assert digest(out/'models.json')==frozen['models_sha256'];checks+=1
    models=read(out/'models.json');pairs=read('configs/evidence_graph_dataset_v1.json')['pairs'];rows=[];newseeds=[]
    for shard in sorted(out.glob('shard_*')):
        manifest=read(shard/'manifest.json');assert not manifest['exclusions']
        for scene in manifest['scenes']:
            newseeds.append(scene['seed'])
            root=shard/'scenes'/f'{scene["scene"]:04d}'
            for file,sha in scene['hashes'].items():assert digest(root/file)==sha;checks+=1
        for file,sha in manifest['outputs'].items():assert digest(shard/file)==sha;checks+=1
        for row in lines(shard/'metrics.jsonl'):
            sid=row['scene'];method=row['method'];root=shard/'scenes'/f'{sid:04d}'
            episodefile=shard/'episodes'/f'{sid:04d}_{method}.json';ep=read(episodefile)
            assert digest(episodefile)==row['episode_sha256'];checks+=1
            assert evaluate(root,ep['prediction'])==row['metric'];checks+=1
            assert ep['cost']['returned'] and ep['cost']['closed'];checks+=1
            assert ep['cost']['history_views']<=4 and ep['cost']['additional_current_views']<=2;checks+=1
            assert abs(sum(e['travel_m'] for e in ep['events'])-ep['cost']['travel_m'])<1e-10;checks+=1
            events=[e for e in ep['events'] if e.get('epoch')]
            assert len(events)==ep['backend_fits']+1;checks+=1
            for stage,r in enumerate(ep['trajectory']):
                saved=read_graph(shard/'snapshots'/f'{sid:04d}_{method}_{stage}.json.gz')
                graph=saved['graph'];sources={e['epoch']+'_'+f"{e['view_id']:02d}" for e in events[:stage+2]}
                assert set(graph['audit']['source_tokens'])==sources;checks+=1
                assert saved['decision']==r;checks+=1
                with patch('builtins.open',side_effect=AssertionError()),patch('io.open',side_effect=AssertionError()),patch('os.open',side_effect=AssertionError()):
                    rep=representation(graph,pairs,pooled=method=='pooled_rank')
                    if method in models:
                        choice,details=decide(rep,models[method])
                        assert details==r['details'];checks+=1
                    elif method=='coverage_replan' or stage==0:choice=int(np.argmax(rep['coverage']))
                    else:
                        plan=ep['trajectory'][0]['selected_option']
                        remaining=plan[stage:] if stage<len(plan) else []
                        # Only next committed single is shown in later snapshot.
                        option=tuple(map(tuple,remaining[:1]));choice=rep['options'].index(option)
                    assert list(map(list,rep['options'][choice]))==r['selected_option'];checks+=1
                if r['selected_option']:
                    action=r['selected_option'][0];event=events[stage+2]
                    assert [event['epoch'],event['view_id']]==action;checks+=1
                    assert not any([e['epoch'],e['view_id']]==action for e in events[:stage+2]);checks+=1
            rows.append(row)
    assert len(newseeds)==32 and len(set(newseeds))==32 and min(newseeds)==cfg['seed_start'];checks+=1
    assert len(rows)==32*len(cfg['methods']);checks+=1
    # Four chosen models retrained only from frozen train groups; exact weights.
    groups,_=training_groups(cfg,pairs);retrain={}
    for method,model in models.items():
        with patch('builtins.open',side_effect=AssertionError()),patch('io.open',side_effect=AssertionError()),patch('os.open',side_effect=AssertionError()):
            again=fit(groups['pipeline_train'],method,model['alpha'],cfg['price'])
        assert again==model;checks+=1;retrain[method]='exact'
    # Prespecified fresh scene0 candidate full acquisition/backend replay.
    original=read(out/'shard_0/episodes/0000_set_rank.json');snapshots=[]
    replay=run_episode(out/'shard_0/scenes/0000','set_rank',models,pairs,cfg['budget'],lambda g,p,r:snapshots.append((g,p,r)))
    assert replay['prediction']==original['prediction'];checks+=1
    assert canonical_bytes(replay['trajectory'])==canonical_bytes(original['trajectory']);checks+=1
    for k in ('travel_m','payload_bytes','history_views','additional_current_views'):assert replay['cost'][k]==original['cost'][k];checks+=1
    for stage,(g,p,r) in enumerate(snapshots):
        s=read_graph(out/'shard_0/snapshots'/f'0000_set_rank_{stage}.json.gz')
        assert g==s['graph'] and p==s['prediction'];checks+=1
    summaries={};contrasts={};rng=np.random.default_rng(cfg['bootstrap']['seed'])
    for condition in ('control','shift','all'):
        subset=[r for r in rows if condition=='all' or r['shift']==(condition=='shift')]
        summaries[condition]={m:summarize([r for r in subset if r['method']==m],cfg['price']) for m in cfg['methods']}
        by={(r['scene'],r['method']):r for r in subset};ids=sorted({r['scene'] for r in subset});n=len(ids)
        if condition=='all':
            boot=np.c_[rng.integers(0,16,size=(cfg['bootstrap']['replicates'],16)),rng.integers(16,32,size=(cfg['bootstrap']['replicates'],16))]
        else:boot=rng.integers(0,n,size=(cfg['bootstrap']['replicates'],n))
        contrasts[condition]={}
        for method in cfg['methods']:
            if method=='set_rank':continue
            diff=[];lossdiff=[];cases=[]
            quality_cost={k:[] for k in ('qualified_tp','fp','bad_motion_reports','travel_m','acquisitions')}
            for sid in ids:
                a,b=by[(sid,'set_rank')],by[(sid,method)]
                na=a['cost']['history_views']-1+a['cost']['additional_current_views'];nb=b['cost']['history_views']-1+b['cost']['additional_current_views']
                dl=a['metric']['task_loss']-b['metric']['task_loss'];d=dl+cfg['price']*(na-nb)
                lossdiff.append(dl);diff.append(d);cases.append(dict(scene=sid,objective_difference=d,loss_difference=dl))
                for key in ('qualified_tp','fp','bad_motion_reports'):quality_cost[key].append(a['metric'][key]-b['metric'][key])
                quality_cost['travel_m'].append(a['cost']['travel_m']-b['cost']['travel_m'])
                quality_cost['acquisitions'].append(na-nb)
            d=np.array(diff);dl=np.array(lossdiff)
            contrasts[condition][method]=dict(mean_objective_difference=float(d.mean()),objective_ci95=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),
                   mean_loss_difference=float(dl.mean()),loss_ci95=np.quantile(dl[boot].mean(1),[.025,.975]).tolist(),
                   wins=int(sum(d< -1e-10)),ties=int(sum(abs(d)<=1e-10)),losses=int(sum(d>1e-10)),cases=cases)
            contrasts[condition][method]['quality_cost_differences']={k:dict(mean_per_family=float(np.mean(v)),
                ci95=np.quantile(np.asarray(v)[boot].mean(1),[.025,.975]).tolist()) for k,v in quality_cost.items()}
    save(out/'analysis.json',dict(summaries=summaries,contrasts=contrasts,episodes=len(rows),new_backend_fits=sum(r['fits'] for r in rows),
         extra_audit_replay_fits=replay['backend_fits'],failed_audit_replay_fits=read(out/'audit_first_attempt_failure.json')['completed_replay_backend_fits'] if (out/'audit_first_attempt_failure.json').exists() else 0,
         checks=checks,exact_retrained_models=retrain,new_families=32,new_payloads=32*24,
         zero_exclusions=True,scope='actual closed-loop, new seeds; same ideal palette generator and translation-only components',
         inference='exploratory, candidate designed after old DEV;32freshfamilies16percondition; no multiple-testing confirmation',
         data_support='training only initial/one-action prefixes; later states explicitly extrapolation'))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','method']+list(summaries['all']['set_rank']));writer.writeheader()
        for cond,methods in summaries.items():
            for method,v in methods.items():writer.writerow(dict(condition=cond,method=method,**v))
    print('PASS',checks,'checks;',len(rows),'real episodes;',sum(r['fits'] for r in rows),'new fits')
    for cond in summaries:
        print(cond,{k:(v['qualified_tp'],v['fp'],round(v['task_loss'],3),round(v['mean_acquisitions'],3)) for k,v in summaries[cond].items()},flush=True)


if __name__=='__main__':main()
