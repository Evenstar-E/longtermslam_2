"""Frozen-prefix stop/cost audit. No new policies or latent-data acquisition."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from inspection_decisions.costs import prefix_cost,report_risk,threshold_stop
from comparative_evidence.evaluation_v3 import evaluate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfgpath=Path('configs/inspection_stopping_audit_v1.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/inspection_stopping_audit_v1b')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    manifest=dict(inputs={},sources={str(p):sha(p) for p in [cfgpath,Path(__file__),Path('inspection_decisions/costs.py')]},exclusions=[])
    checks=0;prefixes=[];rules=[]
    for shard in range(2):
        m=json.loads((Path(cfg['source'])/f'shard_{shard}'/'manifest.json').read_text())
        for p,h in m['sources'].items():assert sha(p)==h;checks+=1
    for sid in range(48):
        scene=Path('results/comparative_interface_v1/scenes')/f'{sid:04d}';cat=scene/'catalog.json'
        meta=json.loads(cat.read_text());manifest['inputs'][str(cat)]=sha(cat)
        for method in cfg['methods']:
            source=Path(cfg['source'])/f'shard_{sid%2}'/'predictions'/f'{sid:04d}_{method}.json'
            manifest['inputs'][str(source)]=sha(source);r=json.loads(source.read_text());snap=r['prediction']['snapshots']
            assert len(snap)==6;checks+=1;trace=[]
            for stage,s in enumerate(snap):
                p=s['prediction'];cost=prefix_cost(r['state']['events'],meta,stage,cfg['nominal_speed_m_s'],cfg['nominal_bandwidth_bytes_s'],r['state']['index_bytes'])
                row=dict(scene=sid,method=method,stage=stage,mean_declared_risk=report_risk(p),
                    metric=evaluate(scene,p),cost=cost,speed_sensitivity={str(v):prefix_cost(r['state']['events'],meta,stage,v,cfg['nominal_bandwidth_bytes_s'],r['state']['index_bytes'])['modeled_motion_sensor_transfer_s'] for v in cfg['speed_sensitivity_m_s']})
                if stage==5:
                    assert abs(cost['total_travel_m']-r['cost']['travel_m'])<1e-10;assert cost['payload_bytes']==r['cost']['payload_bytes'];checks+=2
                trace.append(row);prefixes.append(row)
            # Evaluations are held outside the baseline stopper: it receives
            # predictions only, not metric fields or future outcomes.
            for stage in cfg['stages']:rules.append(dict(rule=f'fixed_{stage}',**trace[stage]))
            for threshold in cfg['risk_thresholds_mean_declared_loss']:
                stage=threshold_stop((s['prediction'] for s in snap),threshold)
                rules.append(dict(rule=f'risk_{threshold:g}',**trace[stage]))
        print(f'prefix audit scene{sid:02d}',flush=True)
    summaries={}
    for method in cfg['methods']:
        summaries[method]={}
        for rule in dict.fromkeys(r['rule'] for r in rules):
            rr=[r for r in rules if r['method']==method and r['rule']==rule];assert len(rr)==48;checks+=1
            summaries[method][rule]=dict(**{k:sum(r['metric'][k] for r in rr) for k in ('qualified_tp','fp','bad_motion_reports','known','task_loss')},
                mean_stage=float(np.mean([r['stage'] for r in rr])),
                mean_travel_m=float(np.mean([r['cost']['total_travel_m'] for r in rr])),
                mean_history=float(np.mean([r['cost']['history_views'] for r in rr])),
                mean_extra_current=float(np.mean([r['cost']['additional_current_views'] for r in rr])),
                mean_payload_bytes=float(np.mean([r['cost']['payload_bytes'] for r in rr])),
                mean_modeled_seconds=float(np.mean([r['cost']['modeled_motion_sensor_transfer_s'] for r in rr])),
                mean_inference_seconds=None,
                speed_sensitivity_seconds={str(v):float(np.mean([r['speed_sensitivity'][str(v)] for r in rr])) for v in cfg['speed_sensitivity_m_s']})
    # All adjacent prefixes, not only changes in a threshold's selected cases.
    transitions=[]
    for method in cfg['methods']:
        for sid in range(48):
            trace=[r for r in prefixes if r['method']==method and r['scene']==sid]
            for before,after in zip(trace,trace[1:]):
                transitions.append(dict(scene=sid,method=method,stage=after['stage'],
                    model_risk_change=after['mean_declared_risk']-before['mean_declared_risk'],
                    actual_loss_change=after['metric']['task_loss']-before['metric']['task_loss'],
                    qualified_change=after['metric']['qualified_tp']-before['metric']['qualified_tp']))
    (out/'prefixes.jsonl').write_text('\n'.join(json.dumps(r) for r in prefixes)+'\n',encoding='utf-8')
    (out/'rules.jsonl').write_text('\n'.join(json.dumps(r) for r in rules)+'\n',encoding='utf-8')
    (out/'transitions.jsonl').write_text('\n'.join(json.dumps(r) for r in transitions)+'\n',encoding='utf-8')
    result=dict(scope='192REUSEDtraces1152prefixes2304ruleevaluations;nonewpolicyexecution/fits/scenes,exploratoryofflinecausalprefixruleaudit',
        summaries=summaries,checks=checks,exclusions=[],
        nonmonotonic=dict(adjacent_prefixes=len(transitions),
            actual_loss_increases=sum(r['actual_loss_change']>1e-10 for r in transitions),
            qualified_decreases=sum(r['qualified_change']<0 for r in transitions),
            model_risk_down_but_actual_loss_up=sum(r['model_risk_change']< -1e-10 and r['actual_loss_change']>1e-10 for r in transitions)),
        limits=['No threshold selected for deployment','No calibrated risk/error bound','No training/test split: all reusedDEV',
                'PrefixCPU unavailable; modeledseconds are not total latency or measured UAV time'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fields=['method','rule','qualified_tp','fp','bad_motion_reports','known','task_loss','mean_stage','mean_travel_m','mean_history','mean_extra_current','mean_payload_bytes','mean_modeled_seconds']
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method,dd in summaries.items():
            for rule,r in dd.items():writer.writerow(dict(method=method,rule=rule,**{k:r[k] for k in fields[2:]}))
    manifest['outputs']={p.name:sha(p) for p in out.iterdir() if p.is_file()};manifest['checks']=checks
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
