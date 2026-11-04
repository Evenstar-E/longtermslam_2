"""Audit every execution and evaluate paired scene-level development effects."""
import hashlib,json,sys
import scipy
from collections import defaultdict
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v5 import run_selection_v5


def aggregate(rows):
    keys=('qualified_tp','tp','fp','bad_motion_reports','known','wrong_known','changed','assets','task_loss','identity_wrong_references')
    result={k:sum(r[k] for r in rows) for k in keys}
    result['mean_travel_m']=float(np.mean([r['cost']['travel_m'] for r in rows]))
    result['median_policy_compute_s']=float(np.median([r['timings']['policy_wall_excluding_api_s'] for r in rows]))
    return result


def main():
    base=Path('results/comparative_selection_v5');cfg=json.loads(Path('configs/comparative_selection_v5.json').read_text())
    plan=json.loads(Path('configs/comparative_selection_v5_analysis.json').read_text())
    checks=0;rows=[];saved_cache={};manifests=[]
    for shard in range(2):
        folder=base/f'shard_{shard}';m=json.loads((folder/'manifest.json').read_text());manifests.append(m)
        assert m['executions']==168 and m['completed']==504 and not m['exclusions'];checks+=1
        for p,digest in m['sources'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest;checks+=1
        assert (folder/'frozen_config.json').read_bytes()==Path('configs/comparative_selection_v5.json').read_bytes();checks+=1
        for row in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines()):
            key=row['scene'],row['method']
            if key not in saved_cache:
                p=folder/'predictions'/f"{row['scene']:04d}_{row['method']}.json";raw=p.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==row['prediction_sha256'];checks+=1
                saved_cache[key]=json.loads(raw)
                saved=saved_cache[key]
                assert saved['cost']['history_views']==4 and saved['cost']['additional_current_views']==2
                assert saved['cost']['returned'] and saved['state']['closed'];checks+=1
                known={'history':{0},'current':{0}}
                for d in saved['prediction']['decisions']:
                    alternatives=d['alternatives']
                    assert d['score']==max(a['score'] for a in alternatives);checks+=1
                    assert len(alternatives)==len({(a['epoch'],a['view']) for a in alternatives});checks+=1
                    for a in alternatives:
                        assert a['view'] not in known[a['epoch']]
                        assert len(known[a['epoch']])<(4 if a['epoch']=='history' else 3);checks+=1
                        opposite='current' if a['epoch']=='history' else 'history'
                        assert set(a['source_tokens'])<= {f'{opposite}_{i:02d}' for i in known[opposite]};checks+=1
                        assert 0<=a['mean_missing_support']<=1+1e-12;checks+=1
                    assert d['view'] not in known[d['epoch']];known[d['epoch']].add(d['view']);checks+=1
                assert len(saved['prediction']['decisions'])==5;checks+=1
                for mode,audit in zip(saved['prediction']['mode_audit'],saved['prediction']['uncertainty_audit']):
                    assert abs(mode['energy']['data']-audit['data_energy'])<1e-7;checks+=1
            saved=saved_cache[key];p=saved['prediction']
            pred=p if row['variant']=='risk' else p['terminal_alternatives'][row['variant']]
            score=evaluate(Path(cfg['source_data'])/f"{row['scene']:04d}",pred)
            assert all(row[k]==v for k,v in score.items());checks+=1
            rows.append(row)
    assert len(saved_cache)==336 and len(rows)==1008;checks+=1
    for path,field in [('results/comparative_active_v3/manifest.json','sources'),
        ('results/comparative_reports_v4/manifest.json','sources'),
        ('results/comparative_selection_v2/manifest.json','source_sha256'),
        ('results/comparative_selection_v1/manifest.json','source_sha256')]:
        for p,digest in json.loads(Path(path).read_text())[field].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest;checks+=1
    data=json.loads(Path('results/comparative_interface_v1/manifest.json').read_text())
    cells=defaultdict(list);conditions={}
    for scene in data['scenes']:
        root=Path(cfg['source_data'])/f"{scene['scene']:04d}"
        for p,digest in scene['hashes'].items():
            assert hashlib.sha256((root/p).read_bytes()).hexdigest()==digest;checks+=1
        cond=scene['condition'];conditions[scene['scene']]=cond
        cells[json.dumps(cond,sort_keys=True)].append(scene['scene'])
    assert len(cells)==16 and all(len(v)==3 for v in cells.values());checks+=1
    groups={(m,v):sorted([r for r in rows if r['method']==m and r['variant']==v],key=lambda r:r['scene'])
            for m in cfg['methods'] for v in ('risk','point','two_sigma')}
    summaries={m:{v:aggregate(groups[m,v]) for v in ('risk','point','two_sigma')} for m in cfg['methods']}
    rng=np.random.default_rng(plan['seed'])
    draws=np.stack([np.concatenate([np.array(c)[rng.integers(0,3,3)] for c in cells.values()]) for _ in range(plan['resamples'])])
    pairs=[tuple(s.split(' vs ')) for s in cfg['comparisons']]+[tuple(p) for p in plan['secondary_exploratory_pairs']]
    comparisons={}
    for left,right in pairs:
        a=groups[left,'risk'];b=groups[right,'risk']
        delta=np.array([x['qualified_tp']-y['qualified_tp'] for x,y in zip(a,b)])
        changed=np.array([x['changed'] for x in a]);den=changed[draws].sum(1)
        distribution=delta[draws].sum(1)/den
        comparisons[left+' minus '+right]=dict(qualified_difference=int(delta.sum()),
            difference_pp=float(delta.sum()/changed.sum()*100),
            exploratory_95_ci_pp=(np.quantile(distribution,[.025,.975])*100).tolist(),
            scene_differences=delta.tolist())
    geometry={};actions={};factor={}
    for method in cfg['methods']:
        geo=defaultdict(int);action=defaultdict(float)
        for sid in range(48):
            saved=saved_cache[sid,method];p=saved['prediction'];r=groups[method,'risk'][sid]
            truth=json.loads((Path(cfg['source_data'])/f'{sid:04d}'/'evaluation/truth.json').read_text())
            for own in r['ownership_records']:
                owner=own['owner']
                if owner<0 or not own['identity_ok'] or not truth['changed'][owner]:continue
                j=next(j for j,a in enumerate(p['assets']) if a['ref_id']==own['ref_id'])
                err=np.linalg.norm(np.array(p['assets'][j]['point_estimate_delta'])-truth['delta'][owner])
                if err>.10:
                    geo['point_bad']+=1
                    errors=[np.linalg.norm(np.array(m['mean_delta'][j])-truth['delta'][owner]) for m in p['mode_audit']]
                    geo['other_mode_good']+=int(min(errors)<=.10)
                    geo['all_modes_bad']+=int(min(errors)>.10)
            for stage,d in enumerate(p['decisions']):
                action['history_actions']+=d['epoch']=='history';action['current_actions']+=d['epoch']=='current'
                chosen=next(a for a in d['alternatives'] if a['epoch']==d['epoch'] and a['view']==d['view'])
                action['chosen_missing_support_sum']+=chosen['mean_missing_support']
                action['selected_pair_rows_sum']+=chosen['pair_rows']
                action['selected_zero_or_negative_gain']+=chosen['gain']<=1e-12
                if method=='support_candidate':
                    old=saved_cache[sid,'legacy_candidate']['prediction']['decisions'][stage]
                    action['same_stage_action_as_legacy_candidate']+=(d['epoch'],d['view'])==(old['epoch'],old['view'])
        geometry[method]=dict(geo);actions[method]=dict(action)
        factor[method]={}
        for key in ('limited_history','coherent_error','repeated_appearance'):
            factor[method][key]={str(value):aggregate([r for r in groups[method,'risk'] if conditions[r['scene']][key]==value])
                                 for value in (False,True)}
    # Prespecified scene1 candidate replay; independent of failure selection.
    replay=execute_policy(Path(cfg['source_data'])/'0001',*cfg['budget'],lambda port:run_selection_v5(port,'support_candidate'))
    assert replay['prediction']==saved_cache[1,'support_candidate']['prediction'];checks+=1
    result=dict(scope='48 reused DEV bases,336 actual policies,1008 terminal evaluations; no independent confirmation',
        checks=checks,summaries=summaries,comparisons=comparisons,geometry=geometry,actions=actions,
        factors=factor,exclusions=[],shard_seconds=[m['seconds'] for m in manifests],
        multiplicity='8 exploratory contrasts, not familywise confirmation',bootstrap=plan,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for m in cfg['methods']:print(m,json.dumps(summaries[m]['risk']))
    print('comparisons',json.dumps(comparisons));print('geometry',json.dumps(geometry));print('checks',checks)


if __name__=='__main__':main()
