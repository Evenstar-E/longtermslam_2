"""Audit every graph/label and replay prespecified scene0. No model fitting to win."""
import collections,gzip,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from comparative_evidence.interface import Episode
from inspection_decisions.graph_dataset import collect_scene,digest,canonical_bytes,read_graph
from inspection_decisions.evidence_graph import NODE_FEATURES,EDGE_FEATURES,without_nuisance_edges

CHECKS=0


def check(x):
    global CHECKS
    assert x
    CHECKS+=1


def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]


def main():
    base=Path('results/evidence_graph_dataset_v1');cfg=json.loads(Path('configs/evidence_graph_dataset_v1.json').read_text())
    states={};labels={};transitions=[];tl={};pairs=[];scenes=[];manifests=[];statistics=[];sourceblocks=[]
    begin=time.perf_counter()
    for shard in range(2):
        out=base/f'shard_{shard}';m=json.loads((out/'manifest.json').read_text());manifests.append(m)
        check(not m['exclusions']);check(m['states']==744 and m['solve_count']==744 and m['transitions']==1464 and m['pairs']==192)
        check((out/'frozen_config.json').read_bytes()==Path('configs/evidence_graph_dataset_v1.json').read_bytes())
        for p,h in m['sources'].items():check(digest(p)==h)
        for p,h in m['outputs'].items():check(digest(out/p)==h)
        for ss in m['scenes']:
            root=out/'scenes'/f"{ss['scene']:04d}";scenes.append((ss,root))
            for path,h in ss['hashes'].items():check(digest(root/path)==h)
        for s in rows(out/'index/states.jsonl'):
            check(s['state_id'] not in states);states[s['state_id']]=s
        for r in rows(out/'labels/state_outcomes.jsonl'):check(r['state_id'] not in labels);labels[r['state_id']]=r
        transitions+=rows(out/'inputs/transitions.jsonl');pairs+=rows(out/'labels/pair_interactions.jsonl')
        for r in rows(out/'labels/transition_outcomes.jsonl'):check(r['transition_id'] not in tl);tl[r['transition_id']]=r
    check(len(states)==1488 and len(labels)==1488 and len(transitions)==2928 and len(pairs)==384)
    scene_lookup={s['scene']:(s,root) for s,root in scenes};family_split={};schemas=set();graph_cache={}
    for key,s in states.items():
        ss,root=scene_lookup[s['scene']];check(s['family_id']==ss['family_id'] and s['split']==ss['split'])
        if s['family_id'] in family_split:check(family_split[s['family_id']]==s['split'])
        family_split[s['family_id']]=s['split']
        check(digest(s['input_file'])==s['input_sha256']);check(digest(s['estimator_file'])==s['estimator_sha256'])
        raw=gzip.decompress(Path(s['input_file']).read_bytes());check(hashlib.sha256(raw).hexdigest()==s['uncompressed_sha256'])
        g=json.loads(raw);schemas.add(g['schema']);graph_cache[key]=g
        check(set(g)=={'schema','node_feature_names','edge_feature_names','nodes','edges','actions','context','audit'})
        check(g['node_feature_names']==NODE_FEATURES and g['edge_feature_names']==EDGE_FEATURES)
        p=json.loads(Path(s['estimator_file']).read_text());check(evaluate(root,p)==labels[key]['metric'])
        known={'history':{0},'current':{0}}
        for epoch,view in s['actions']:
            check(view not in known[epoch]);known[epoch].add(view)
        expected_tokens={f'{e}_{v:02d}' for e,vs in known.items() for v in vs}
        check(set(g['audit']['source_tokens'])==expected_tokens and set(s['source_payloads'])==expected_tokens)
        for token,h in s['source_payloads'].items():check(digest(root/'payloads'/f'{token}.npz')==h)
        expected={('stop',-1)}
        if len(known['history'])<4:expected|={('history',v) for v in range(12) if v not in known['history']}
        if len(known['current'])-1<2:expected|={('current',v) for v in range(12) if v not in known['current']}
        check({(a['epoch'],a['view']) for a in g['actions']}==expected)
        nodes={n['id']:n for n in g['nodes']};check(set(nodes)==set(range(len(nodes))))
        for node in nodes.values():
            check(node['kind'] in NODE_FEATURES);check(len(node['features'])==len(NODE_FEATURES[node['kind']]))
            check(bool(np.isfinite(node['features']).all()))
        counts=collections.Counter()
        for edge in g['edges']:
            check(edge['source'] in nodes and edge['target'] in nodes);check(edge['kind'] in EDGE_FEATURES)
            check(len(edge['features'])==len(EDGE_FEATURES[edge['kind']]));check(bool(np.isfinite(edge['features']).all()))
            counts[edge['kind']]+=1
        audit_patch={r['node']:r for r in g['audit']['patch_sources']}
        across=[e for e in g['edges'] if e['kind']=='nuisance_coupling' and audit_patch[e['source']]['epoch']==audit_patch[e['target']]['epoch'] and audit_patch[e['source']]['token']!=audit_patch[e['target']]['token']]
        if across:
            coherent=ss['condition']['coherent_error'];has_shared=any(e['features'][-1]>0 for e in across)
            nonzero=any(np.linalg.norm(e['features'][:9])>1e-12 for e in across)
            check(has_shared==coherent);check(nonzero==coherent)
            sourceblocks.append(dict(state_id=key,coherent=coherent,shared_cross_frame_edges=sum(e['features'][-1]>0 for e in across)))
        for nu in g['audit']['nuisance_keys']:
            if not nu['observed']:
                f=nodes[nu['node']]['features'];check(np.allclose(f[2:9],0) and np.allclose(f[9:],1/3))
        # Every cross-epoch support edge is same anonymous observed reference.
        for edge in g['edges']:
            if edge['kind']=='cross_epoch_support':
                a,b=audit_patch[edge['source']],audit_patch[edge['target']]
                check(a['anonymous_reference']==b['anonymous_reference'] and a['epoch']!=b['epoch'])
        ab=without_nuisance_edges(g);check(ab['nodes']==g['nodes'] and ab['actions']==g['actions'])
        statistics.append(dict(state_id=key,split=s['split'],nodes=len(nodes),edges=len(g['edges']),edge_counts=dict(counts),
            compressed_bytes=Path(s['input_file']).stat().st_size,backend_seconds=s['actual_backend_compute_s']))
    check(schemas=={'observed_evidence_graph_v1'});check(len(family_split)==48)
    old_seed_set={s['seed'] for s in json.loads(Path('results/comparative_interface_v1/manifest.json').read_text())['scenes']}
    check(not ({ss['seed'] for ss,root in scenes}&old_seed_set))
    splits=collections.Counter(family_split.values());check(set(splits.values())=={16})
    for split in cfg['splits']:
        subset=[ss for ss,root in scenes if ss['split']==split]
        check(len({json.dumps(ss['condition'],sort_keys=True) for ss in subset})==16)
    for t in transitions:
        check(t['prestate_id'] in states and t['poststate_id'] in states and t['transition_id'] in tl)
        pre,post=states[t['prestate_id']],states[t['poststate_id']]
        check(pre['family_id']==post['family_id']==t['family_id']);check(pre['split']==post['split']==t['split'])
        ga=graph_cache[t['prestate_id']];check(any(a['node']==t['action_node'] and [a['epoch'],a['view']]==t['action'] for a in ga['actions']))
        pl,ql=labels[t['prestate_id']],labels[t['poststate_id']];target=tl[t['transition_id']]
        check(target['stop_loss_before']==pl['metric']['task_loss'] and target['stop_loss_after']==ql['metric']['task_loss'])
        check(target['realized_loss_drop']==pl['metric']['task_loss']-ql['metric']['task_loss'])
        for k,v in target['actual_resource_increment'].items():check(v==ql['cost'][k]-pl['cost'][k] and v>=-1e-9)
        if t['kind']=='stop':check(pre['state_id']==post['state_id'])
        else:check(post['actions']==pre['actions']+[t['action']])
    for r in pairs:
        sid=r['initial_state'].split('_')[0];a,b=r['actions']
        tag=lambda x:('h' if x[0]=='history' else 'c')+f'{x[1]:02d}'
        L0=labels[r['initial_state']]['metric']['task_loss'];La=labels[f'{sid}_{tag(a)}']['metric']['task_loss']
        Lb=labels[f'{sid}_{tag(b)}']['metric']['task_loss'];Lab=labels[r['endpoint_state']]['metric']['task_loss']
        check(r['realized_interaction']==La+Lb-Lab-L0)
    # Prespecified full new scene0 replay, no new world generation. Geometry,
    # graph bytes, report and labels must all reproduce. Timing is excluded.
    replay=Path(tempfile.mkdtemp(prefix='graph_replay_',dir='.cache'))
    for d in ('inputs/states','estimator_outputs'):(replay/d).mkdir(parents=True)
    ss,root=scene_lookup[0];rr=collect_scene(root,replay,0,ss['split'],ss['family_id'],cfg['pairs'],cfg['budget'])
    for row in rr['states']:
        old=states[row['state_id']];check(row['uncompressed_sha256']==old['uncompressed_sha256'])
        check(Path(row['estimator_file']).read_bytes()==Path(old['estimator_file']).read_bytes())
    for row in rr['state_labels']:check(row==labels[row['state_id']])
    for row in rr['transition_labels']:check(row==tl[row['transition_id']])
    for version in ('comparative_selection_v8','inspection_stopping_audit_v1b'):
        paths=list(Path('results',version).glob('shard_*/manifest.json')) if version.endswith('v8') else [Path('results',version,'manifest.json')]
        for path in paths:
            m=json.loads(path.read_text())
            for p,h in m['sources'].items():check(digest(p)==h)
            if 'data' in m:
                for p,h in m['data'].items():check(digest(p)==h)
    inter=np.array([r['realized_interaction'] for r in pairs]);split_summary={}
    for split in cfg['splits']:
        ix=[r for r in pairs if r['split']==split];v=np.array([r['realized_interaction'] for r in ix])
        split_summary[split]=dict(families=16,states=sum(s['split']==split for s in states.values()),pairs=len(ix),
            positive=int((v>1e-9).sum()),negative=int((v< -1e-9).sum()),zero=int((abs(v)<=1e-9).sum()))
    result=dict(scope='48NEWpipelinefamilies/1152freshRGBDpayloads/1488actualestimatorfits;notlearnedpolicyorindependentconfirmation',
        counts=dict(families=48,states=len(states),transitions=len(transitions),pairs=len(pairs),actual_estimator_fits=sum(m['solve_count'] for m in manifests),
            actual_payload_reads=sum(m['actual_payload_reads'] for m in manifests),fresh_payload_files=1152),
        splits=split_summary,label_census=dict(zero_loss_states=sum(l['metric']['task_loss']==0 for l in labels.values()),
            identity_wrong_references_across_all_states=sum(l['metric']['identity_wrong_references'] for l in labels.values()),
            caution='Repeated state counts,notindependentcases;allstateskept'),graph_statistics=dict(mean_nodes=float(np.mean([s['nodes'] for s in statistics])),
            mean_edges=float(np.mean([s['edges'] for s in statistics])),total_compressed_bytes=sum(s['compressed_bytes'] for s in statistics)),
        realized_interactions=dict(positive=int((inter>1e-9).sum()),negative=int((inter< -1e-9).sum()),zero=int((abs(inter)<=1e-9).sum()),
            mean=float(inter.mean()),range=[float(inter.min()),float(inter.max())],
            caution='Repeated branches correlated within48families;one realized outcome,notexpected/causal/optimalpairgain oralgorithmbenefit'),
        shared_nuisance_structure=dict(states_with_cross_frame_pairs=len(sourceblocks),
            coherent_nonzero_states=sum(r['coherent'] for r in sourceblocks),independent_zero_states=sum(not r['coherent'] for r in sourceblocks),
            caution='Structural expressibility check,notutility/performanceevidence'),
        checks=CHECKS,exclusions=[],seconds=[m['seconds'] for m in manifests],analysis_seconds=time.perf_counter()-begin,
        full_replay_scene=0,analyzer_sha256=digest(__file__),statistics=statistics)
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='statistics'},ensure_ascii=False))


if __name__=='__main__':main()
