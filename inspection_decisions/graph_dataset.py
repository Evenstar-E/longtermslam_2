"""Truth-separated collector orchestration. Graph inputs persist before labels.

Only this orchestrator/evaluator can access the scene directory. The graph
builder gets acquired observation objects and whitelisted estimated state only.
"""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_terminal
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.evidence_graph import build_graph
from inspection_decisions.costs import prefix_cost


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def write_json(path,value,compressed=False):
    raw=canonical_bytes(value)
    Path(path).write_bytes(gzip.compress(raw,mtime=0) if compressed else raw)
    return hashlib.sha256(raw).hexdigest()


def read_graph(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def action_name(action):return ('h' if action[0]=='history' else 'c')+f'{action[1]:02d}'


def collect_scene(scene,out,sid,split,family_id,pairs,budget):
    """Returns separate input/index/label records, never embeds labels in graph."""
    scene=Path(scene);out=Path(out);states=[];transitions=[];state_labels=[];transition_labels=[];interactions=[]
    saved={};solve_count=0;actual_reads=0
    def episode():
        nonlocal actual_reads
        ep=Episode(scene,*budget);h=[ep.act('retrieve_history',0)];q=[ep.preview()]
        actual_reads+=2
        return ep,h,q,ep.metadata()
    def acquire(ep,h,q,action):
        nonlocal actual_reads
        obs=ep.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1]);actual_reads+=1
        (h if action[0]=='history' else q).append(obs)
        return obs
    def snapshot(ep,h,q,meta,name,actions):
        nonlocal solve_count
        begin=time.perf_counter();fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
        p=common_terminal(fit);backend_seconds=time.perf_counter()-begin;solve_count+=1
        graph=build_graph(fit,h,q,meta,ep.state(),p)
        state_id=f'{sid:04d}_{name}';input_path=out/'inputs'/'states'/f'{state_id}.json.gz'
        input_digest=write_json(input_path,graph,True)  # BEFORE evaluator below
        pred_path=out/'estimator_outputs'/f'{state_id}.json';write_json(pred_path,p)
        cost=prefix_cost(ep.state()['events'],meta,len(actions),index_bytes=ep.state()['index_bytes'])
        sources={o['token']:o['provenance_sha256'] for o in h+q}
        record=dict(state_id=state_id,scene=sid,split=split,family_id=family_id,
            input_file=str(input_path),input_sha256=digest(input_path),uncompressed_sha256=input_digest,
            estimator_file=str(pred_path),estimator_sha256=digest(pred_path),
            actions=[list(a) for a in actions],source_payloads=sources,actual_backend_compute_s=backend_seconds)
        states.append(record)
        # Truth appears only in separate label records and never flows into
        # fit_cache, graph construction, or preselected branch selection.
        label=dict(state_id=state_id,metric=evaluate(scene,p),cost=cost)
        state_labels.append(label);saved[name]=(record,label,graph)
        return fit
    def transition(before,after,action,kind):
        br,bl,bg=saved[before];ar,al,ag=saved[after]
        a=next(a for a in bg['actions'] if (a['epoch'],a['view'])==action)
        tid=f"{br['state_id']}__{action_name(action) if action[0]!='stop' else 'stop'}__{ar['state_id']}"
        transitions.append(dict(transition_id=tid,prestate_id=br['state_id'],action_node=a['node'],
            action=list(action),poststate_id=ar['state_id'],kind=kind,split=split,family_id=family_id))
        delta_cost={k:al['cost'][k]-bl['cost'][k] for k in ('total_travel_m','payload_bytes','modeled_acquisition_s','modeled_motion_sensor_transfer_s')}
        if any(v< -1e-9 for v in delta_cost.values()):raise AssertionError('Negative reserved cost increment')
        transition_labels.append(dict(transition_id=tid,stop_loss_before=bl['metric']['task_loss'],
            stop_loss_after=al['metric']['task_loss'],realized_loss_drop=bl['metric']['task_loss']-al['metric']['task_loss'],
            actual_resource_increment=delta_cost,label_kind='realized outcome, not conditional expected value'))
    ep,h,q,meta=episode();fit=snapshot(ep,h,q,meta,'initial',[]);ep.finish()
    initial=saved['initial'][2]
    options=[(a['epoch'],a['view']) for a in initial['actions'] if a['epoch']!='stop']
    # All branch choices are fixed by this observed initial action list/config.
    for action in options:
        ep,h,q,meta=episode();acquire(ep,h,q,action);name=action_name(action)
        snapshot(ep,h,q,meta,name,[action]);ep.finish()
        transition('initial',name,action,'single')
    for aa,bb in pairs:
        first,second=tuple(aa),tuple(bb);ep,h,q,meta=episode();acquire(ep,h,q,first)
        # Reuse the already stored first-prefix INPUT, never its truth label. Verify
        # that every legally read payload equals the earlier prefix sources.
        source={o['token']:o['provenance_sha256'] for o in h+q}
        assert source==saved[action_name(first)][0]['source_payloads']
        acquire(ep,h,q,second);name=action_name(first)+'_'+action_name(second)
        snapshot(ep,h,q,meta,name,[first,second]);ep.finish()
        transition(action_name(first),name,second,'pair_second_step')
        L0=saved['initial'][1]['metric']['task_loss'];La=saved[action_name(first)][1]['metric']['task_loss']
        Lb=saved[action_name(second)][1]['metric']['task_loss'];Lab=saved[name][1]['metric']['task_loss']
        interactions.append(dict(initial_state=saved['initial'][0]['state_id'],actions=[list(first),list(second)],
            endpoint_state=saved[name][0]['state_id'],realized_interaction=La+Lb-Lab-L0,
            formula='L(a)+L(b)-L(ab)-L(initial); one realized branch, not expected synergy',split=split,family_id=family_id))
    for name in list(saved):transition(name,name,('stop',-1),'stop')
    return dict(states=states,transitions=transitions,state_labels=state_labels,
        transition_labels=transition_labels,pair_labels=interactions,solve_count=solve_count,actual_payload_reads=actual_reads)
