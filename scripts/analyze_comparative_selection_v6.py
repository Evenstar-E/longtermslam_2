"""Full immutable-input audit, paired weight contrasts, selected-action diagnostics."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import spearmanr
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v6 import run_selection_v6


def aggregate(rows):
    fields=('qualified_tp','tp','fp','bad_motion_reports','known','changed','assets','task_loss','identity_wrong_references')
    r={k:sum(s[k] for s in rows) for k in fields}
    r['mean_travel_m']=float(np.mean([s['cost']['travel_m'] for s in rows]))
    return r


def correlation(records,x,y):
    pairs=[(r[x],r[y]) for r in records if r[y] is not None]
    a,b=np.array(pairs).T
    rho=None if np.ptp(a)<1e-12 or np.ptp(b)<1e-12 else float(spearmanr(a,b).statistic)
    return dict(descriptive_rho=rho,repeated_selected_steps=len(pairs),
                inference='No independent-step p-value; only selected actions, not unchosen rankings')


def main():
    base=Path('results/comparative_selection_v6');cfg=json.loads(Path('configs/comparative_selection_v6.json').read_text())
    rows=[];snaprows=[];saved_cache={};checks=0;manifests=[]
    for shard in range(2):
        folder=base/f'shard_{shard}';m=json.loads((folder/'manifest.json').read_text());manifests.append(m)
        assert m['new_policies']==m['replays']==48 and m['executions']==96
        assert m['terminal_evaluations']==288 and m['snapshot_evaluations']==1152 and not m['exclusions'];checks+=1
        for group in ('sources','inputs'):
            for p,d in m[group].items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d;checks+=1
        assert (folder/'frozen_config.json').read_bytes()==Path('configs/comparative_selection_v6.json').read_bytes();checks+=1
        for row in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines()):
            sid,method,variant=row['scene'],row['method'],row['variant'];key=sid,method
            root=Path(cfg['source_data'])/f'{sid:04d}'
            if key not in saved_cache:
                target=folder/'predictions'/f'{sid:04d}_{method}.json';raw=target.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==row['prediction_sha256'];checks+=1
                saved=json.loads(raw);saved_cache[key]=saved;p=saved['prediction']
                assert saved['cost']['history_views']==4 and saved['cost']['additional_current_views']==2 and saved['cost']['returned'];checks+=1
                assert len(p['snapshots'])==6 and len(p['decisions'])==5;checks+=1
                known={'history':{0},'current':{0}}
                for stage,d in enumerate(p['decisions']):
                    assert d['score']==max(a['score'] for a in d['alternatives']);checks+=1
                    for a in d['alternatives']:
                        assert a['view'] not in known[a['epoch']];checks+=1
                        opposite='current' if a['epoch']=='history' else 'history'
                        assert set(a['source_tokens'])<= {f'{opposite}_{v:02d}' for v in known[opposite]};checks+=1
                        assert a['pair_weighting']==('uniform' if method.startswith('uniform_') else 'missing_support');checks+=1
                        if method.startswith('uniform_') and a['opposite_visible_points']:
                            assert abs(a['mean_pair_weight']-1)<1e-12;checks+=1
                    known[d['epoch']].add(d['view'])
                assert p['snapshots'][-1]['prediction']['assets']==p['assets'];checks+=1
                if method in cfg['replay_methods']:
                    oldname=cfg['replay_methods'][method]
                    old=json.loads(Path(f'results/comparative_selection_v5/shard_{sid%2}/predictions/{sid:04d}_{oldname}.json').read_text())
                    assert old['prediction']['assets']==p['assets'];checks+=1
                    for d,e in zip(old['prediction']['decisions'],p['decisions']):
                        assert (d['epoch'],d['view'],d['score'])==(e['epoch'],e['view'],e['score']);checks+=1
            saved=saved_cache[key];p=saved['prediction']
            pred=p if variant=='risk' else p['terminal_alternatives'][variant]
            score=evaluate(root,pred)
            assert all(row[k]==v for k,v in score.items());checks+=1
            rows.append(row)
        for row in map(json.loads,(folder/'steps.jsonl').read_text().splitlines()):
            s=saved_cache[row['scene'],row['method']]['prediction']['snapshots'][row['stage']]
            root=Path(cfg['source_data'])/f"{row['scene']:04d}"
            for key,pred in (('risk',s['prediction']),('point',s['point_prediction'])):
                assert evaluate(root,pred)==row[key];checks+=1
            assert row['model_risk32']==s['model_risk32'] and row['terminal_risk2048']==s['terminal_risk2048'];checks+=1
            snaprows.append(row)
    assert len(rows)==576 and len(snaprows)==1152 and len(saved_cache)==192;checks+=1
    # Preserve all five other v5 controls, do not count reuse as new executions.
    for shard in range(2):
        old=Path(f'results/comparative_selection_v5/shard_{shard}')
        for p,d in json.loads((old/'manifest.json').read_text())['sources'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d;checks+=1
        for row in map(json.loads,(old/'metrics.jsonl').read_text().splitlines()):
            sid,method=row['scene'],row['method']
            old_method=method
            if method in ('support_candidate','support_risk'):
                method='weighted_candidate' if method=='support_candidate' else 'weighted_risk'
                new=next(s for s in rows if (s['scene'],s['method'],s['variant'])==(sid,method,row['variant']))
                for k in ('qualified_tp','tp','fp','bad_motion_reports','known','task_loss','ownership_records'):
                    assert new[k]==row[k];checks+=1
                continue
            target=old/'predictions'/f'{sid:04d}_{old_method}.json'
            assert hashlib.sha256(target.read_bytes()).hexdigest()==row['prediction_sha256'];checks+=1
            rows.append(row)
            if (sid,method) not in saved_cache:saved_cache[sid,method]=json.loads(target.read_text())
    assert len(rows)==1296;checks+=1
    data=json.loads(Path('results/comparative_interface_v1/manifest.json').read_text())
    cells=defaultdict(list)
    for s in data['scenes']:
        root=Path(cfg['source_data'])/f"{s['scene']:04d}"
        for p,d in s['hashes'].items():
            assert hashlib.sha256((root/p).read_bytes()).hexdigest()==d;checks+=1
        cells[json.dumps(s['condition'],sort_keys=True)].append(s['scene'])
    assert len(cells)==16 and all(len(c)==3 for c in cells.values());checks+=1
    methods=['legacy_candidate','legacy_risk','legacy_schur','weighted_candidate','weighted_risk',
             'uniform_candidate','uniform_risk','support_coverage','represented_coverage']
    groups={(m,v):sorted([r for r in rows if r['method']==m and r['variant']==v],key=lambda r:r['scene'])
            for m in methods for v in ('risk','point','two_sigma')}
    summaries={m:{v:aggregate(groups[m,v]) for v in ('risk','point','two_sigma')} for m in methods}
    rng=np.random.default_rng(cfg['analysis']['seed']);draws=np.stack([
        np.concatenate([np.array(c)[rng.integers(0,3,3)] for c in cells.values()]) for _ in range(cfg['analysis']['resamples'])])
    comparisons={}
    for left,right in cfg['analysis']['comparisons']:
        a,b=groups[left,'risk'],groups[right,'risk'];delta=np.array([x['qualified_tp']-y['qualified_tp'] for x,y in zip(a,b)])
        denominator=np.array([r['changed'] for r in a]);dist=delta[draws].sum(1)/denominator[draws].sum(1)*100
        comparisons[left+' minus '+right]=dict(qualified_difference=int(delta.sum()),difference_pp=float(delta.sum()/72*100),
            exploratory_95_ci_pp=np.quantile(dist,[.025,.975]).tolist(),scene_differences=delta.tolist())
    outcomes=[];step_summary={}
    snapindex={(r['scene'],r['method'],r['stage']):r for r in snaprows}
    for method in cfg['new_methods']+list(cfg['replay_methods']):
        method_out=[]
        for sid in range(48):
            truth=json.loads((Path(cfg['source_data'])/f'{sid:04d}/evaluation/truth.json').read_text())
            p=saved_cache[sid,method]['prediction']
            def moved_error(row):
                values=[a['motion_error_m'] for a in row['point']['ownership_records']
                    if a['owner']>=0 and a['identity_ok'] and truth['changed'][a['owner']]]
                return float(np.mean(values)) if values and len(values)==sum(truth['changed']) else None
            for stage,d in enumerate(p['decisions']):
                before=snapindex[sid,method,stage];after=snapindex[sid,method,stage+1]
                chosen=next(a for a in d['alternatives'] if a['epoch']==d['epoch'] and a['view']==d['view'])
                a,b=moved_error(before),moved_error(after)
                sb=p['snapshots'][stage]['prediction']['assets'];sa=p['snapshots'][stage+1]['prediction']['assets']
                # decision_risk is pre-support-gate in the frozen terminal.
                # For the actually emitted unknown action the model loss is .35.
                reported_before=sum(.35 if asset['label']<0 else asset['decision_risk'] for asset in sb)
                reported_after=sum(.35 if asset['label']<0 else asset['decision_risk'] for asset in sa)
                record=dict(scene=sid,method=method,stage=stage,epoch=d['epoch'],view=d['view'],
                    raw_predicted_gain=chosen['gain'],normalized_score=d['score'],
                    model_risk32_drop=before['model_risk32']-after['model_risk32'],
                    terminal_risk2048_drop=before['terminal_risk2048']-after['terminal_risk2048'],
                    reported_action_risk2048_drop=reported_before-reported_after,
                    realized_taskloss_drop=before['risk']['task_loss']-after['risk']['task_loss'],
                    moved_point_mean_error_drop_m=None if a is None or b is None else a-b,
                    point_qualified_gain=after['point']['qualified_tp']-before['point']['qualified_tp'],
                    realized_qualified_gain=after['risk']['qualified_tp']-before['risk']['qualified_tp'],
                    false_change_gain=after['risk']['fp']-before['risk']['fp'])
                method_out.append(record);outcomes.append(record)
        step_summary[method]=dict(selected_steps=len(method_out),
            snapshot_identity_wrong_references=sum(r['point']['identity_wrong_references'] for r in snaprows if r['method']==method),
            positive_prediction_actual_loss_worse=sum(r['raw_predicted_gain']>1e-12 and r['realized_taskloss_drop']< -1e-12 for r in method_out),
            actual_loss_improved_steps=sum(r['realized_taskloss_drop']>1e-12 for r in method_out),
            actual_loss_worse_steps=sum(r['realized_taskloss_drop']< -1e-12 for r in method_out),
            mean_model_risk32_drop=float(np.mean([r['model_risk32_drop'] for r in method_out])),
            mean_terminal_risk2048_drop=float(np.mean([r['terminal_risk2048_drop'] for r in method_out])),
            mean_reported_action_risk2048_drop=float(np.mean([r['reported_action_risk2048_drop'] for r in method_out])),
            mean_actual_taskloss_drop=float(np.mean([r['realized_taskloss_drop'] for r in method_out])),
            correlations={key:correlation(method_out,'raw_predicted_gain',key) for key in
                ('model_risk32_drop','terminal_risk2048_drop','reported_action_risk2048_drop','realized_taskloss_drop','moved_point_mean_error_drop_m')})
        initial_loss=sum(r['risk']['task_loss'] for r in snaprows if r['method']==method and r['stage']==0)
        realized_drop=sum(r['realized_taskloss_drop'] for r in method_out)
        final_loss=summaries[method]['risk']['task_loss']
        assert abs(initial_loss-realized_drop-final_loss)<1e-10;checks+=1
        step_summary[method]['loss_telescoping_check']=dict(initial=initial_loss,sum_decreases=realized_drop,final=final_loss)
    with (base/'action_outcomes.jsonl').open('w',encoding='utf-8') as f:
        for r in outcomes:f.write(json.dumps(r)+'\n')
    replay=execute_policy(Path(cfg['source_data'])/'0001',*cfg['budget'],lambda port:run_selection_v6(port,'uniform_candidate'))
    assert replay['prediction']==saved_cache[1,'uniform_candidate']['prediction'];checks+=1
    result=dict(scope='48 reused DEV bases;96 new policies+96 exact diagnostic replays.576 final evaluations+2304 snapshot evaluations, plus retained v5 controls',
        summaries=summaries,comparisons=comparisons,selected_action_diagnostics=step_summary,checks=checks,
        new_policies=96,exact_v5_replays=96,shard_seconds=[m['seconds'] for m in manifests],exclusions=[],
        analysis_plan=cfg['analysis'],multiplicity='6 exploratory contrasts; not familywise confirmation',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for m in methods:print(m,json.dumps(summaries[m]['risk']))
    print('comparisons',json.dumps(comparisons));print('steps',json.dumps(step_summary));print('checks',checks)


if __name__=='__main__':main()
