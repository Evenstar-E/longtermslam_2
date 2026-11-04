"""POST-HOC mechanism ablation: same scorer, ignore STOP until budget exhaustion.

Every family included. Frozen candidate prefix reused after exact payload checks;
only genuinely new continuation states refit. Does not modify primary results.
"""
import time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_terminal
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.evidence_graph import build_graph
from inspection_decisions.component_set_v1 import representation,decide
from inspection_decisions.graph_dataset import read_graph,write_json,digest
from scripts.run_evidence_value_pilot_v1 import read,save,lines


def main():
    cfg=read('configs/component_set_planner_v1.json');out=Path(cfg['output']);dest=out/'posthoc_no_stop'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir();(dest/'snapshots').mkdir()
    save(dest/'prospective_diagnostic_contract.json',dict(status='POST_HOC relative to primary experiment, fixed before this ablation',
        intervention='same frozen candidate scorer; disable STOP only; all32families; same remaining budget and return',
        purpose='distinguish stopping failure from action-selection failure without training or threshold tuning',
        no_exclusions=True,no_primary_replacement=True,source_sha256=digest(__file__),
        model_sha256=digest(out/'models.json'),primary_analysis_sha256=digest(out/'analysis.json')))
    model=read(out/'models.json')['set_rank'];pairs=read('configs/evidence_graph_dataset_v1.json')['pairs'];records=[];newfits=0;begin=time.perf_counter()
    for sid in range(cfg['new_scenes']):
        shard=out/f'shard_{sid%2}';root=shard/'scenes'/f'{sid:04d}';original=read(shard/'episodes'/f'{sid:04d}_set_rank.json')
        ep=Episode(root,*cfg['budget']);h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
        original_events=[e for e in original['events'] if e.get('epoch')]
        for event in original_events[2:]:
            epoch=event['epoch'];obs=ep.act('retrieve_history' if epoch=='history' else 'observe_current',event['view_id'])
            (h if epoch=='history' else q).append(obs)
        for actual,saved in zip(ep.state()['events'],original_events):
            assert actual['payload_sha256']==saved['payload_sha256'] and actual['epoch']==saved['epoch'] and actual['view_id']==saved['view_id']
        stage=len(original['trajectory'])-1
        cached=read_graph(shard/'snapshots'/f'{sid:04d}_set_rank_{stage}.json.gz');graph=cached['graph'];prediction=cached['prediction']
        continued=[];fits=0
        while True:
            rep=representation(graph,pairs);_,detail=decide(rep,model)
            if len(rep['options'])==1:break
            chosen=1+int(np.argmax(detail['scores'][1:]));option=rep['options'][chosen];action=option[0]
            continued.append(dict(stage=stage,chosen=option,score=detail['scores'][chosen],stop_would_win=detail['scores'][chosen]<=0))
            write_json(dest/'snapshots'/f'{sid:04d}_{stage}_pre.json.gz',dict(graph=graph,decision=continued[-1]),True)
            obs=ep.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1]);(h if action[0]=='history' else q).append(obs)
            fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta);prediction=common_terminal(fit)
            graph=build_graph(fit,h,q,meta,ep.state(),prediction);stage+=1;fits+=1
        cost=ep.finish();save(dest/f'{sid:04d}.json',dict(prediction=prediction,cost=cost,continuation=continued,
                        reused_prefix_source_sha256=digest(shard/'episodes'/f'{sid:04d}_set_rank.json'),new_backend_fits=fits))
        before=evaluate(root,original['prediction']);after=evaluate(root,prediction)
        records.append(dict(scene=sid,shift=sid>=16,before=before,after=after,cost=cost,new_backend_fits=fits,
                            source_initial_stop_stage=len(original['trajectory'])-1))
        newfits+=fits
        if sid%4==3:print('no-STOP diagnosis',sid+1,'families',newfits,'new fits',round(time.perf_counter()-begin,1),'s',flush=True)
    summaries={}
    for condition in ('control','shift','all'):
        selected=[r for r in records if condition=='all' or r['shift']==(condition=='shift')]
        v={side:{k:sum(r[side][k] for r in selected) for k in ('task_loss','qualified_tp','fp','bad_motion_reports','known','identity_wrong_references')} for side in ('before','after')}
        v.update(families=len(selected),mean_acquisitions=float(np.mean([r['cost']['history_views']-1+r['cost']['additional_current_views'] for r in selected])),
                 mean_return_travel_m=float(np.mean([r['cost']['travel_m'] for r in selected])),
                 loss_improved=sum(r['after']['task_loss']<r['before']['task_loss']-1e-10 for r in selected),
                 loss_worsened=sum(r['after']['task_loss']>r['before']['task_loss']+1e-10 for r in selected))
        summaries[condition]=v
    save(dest/'analysis.json',dict(status='POST_HOC intervention; no training or model changes',new_backend_fits=newfits,
                                  summaries=summaries,records=records,seconds=time.perf_counter()-begin))
    print('No-STOP result',summaries['all'],flush=True)


if __name__=='__main__':main()
