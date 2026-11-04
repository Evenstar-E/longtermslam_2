"""Descriptive mechanism checks, never used for fitting or threshold selection."""
from pathlib import Path
import numpy as np
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths
from inspection_decisions.report_risk_stopping_v1 import should_stop


def main():
    out=Path('results/report_risk_stopping_v1');a=read(out/'analysis.json');b=read(out/'models.json')
    if (out/'mechanism_diagnostics.json').exists():raise FileExistsError('diagnostics already exist')
    paths=load_paths(out,'evaluation');records=[];blocks=[];increases=[]
    by={(r['scene'],r['method']):r for r in a['rows']}
    for p in paths:
        if p['policy']!='rank':continue
        sid=p['scene'];states=p['observed']['states'];metrics=p['labels_data']['metrics']
        L=np.array([m['task_loss'] for m in metrics]);previous=None
        for state in states:
            t=state['stage']
            stop,detail=should_stop('dual_temporal',state,previous,states[0]['prediction'],b['heads'],b['settings']['dual_temporal'])
            if t<5 and detail['risk']<=b['settings']['dual_temporal'] and detail['gain']>detail['remaining_price']:
                blocks.append(dict(scene=sid,stage=t,**detail,realized_loss=float(L[t]),full_loss=float(L[-1]),
                    realized_full_gain_per_asset=float((L[t]-L[-1])/8),
                    candidate_reaches_state=t<=by[sid,'dual_temporal']['stage']))
            if t<5 and L[t+1]>L[t]+1e-10:
                increases.append(dict(scene=sid,stage=t,loss_before=float(L[t]),loss_after=float(L[t+1]),
                                      action=state['selected_option'][0]))
            previous=state['prediction']
        vals={m:dict(stage=by[sid,m]['stage'],loss=by[sid,m]['metric']['task_loss'],
                       qualified=by[sid,m]['metric']['qualified_tp'],fp=by[sid,m]['metric']['fp'],
                       bad_motion=by[sid,m]['metric']['bad_motion_reports']) for m in
              ('dual_temporal','dual_static','risk_temporal','declared_rank','original_rank','fixed_rank','full_rank')}
        records.append(dict(scene=sid,shift=p['shift'],methods=vals,loss_sequence=L.tolist(),
            actions=[s['selected_option'][0] for s in states[:-1]],
            candidate_decisions=by[sid,'dual_temporal']['decisions']))
    reached=[r for r in blocks if r['candidate_reaches_state']]
    summaries=dict(candidate_minus_declared=dict(
        improves=sum(r['methods']['dual_temporal']['loss']<r['methods']['declared_rank']['loss']-1e-10 for r in records),
        worsens=sum(r['methods']['dual_temporal']['loss']>r['methods']['declared_rank']['loss']+1e-10 for r in records)),
        candidate_premature_vs_full=[r['scene'] for r in records if r['methods']['dual_temporal']['stage']<5 and
            r['methods']['dual_temporal']['loss']>r['methods']['full_rank']['loss']+1e-10],
        rank_step_loss_increases=len(increases),rank_steps=160,
        gain_gate_blocks_on_candidate_prefix=len(reached),
        gate_blocks_with_zero_or_negative_realized_full_gain=sum(r['realized_full_gain_per_asset']<=1e-10 for r in reached),
        gate_blocks_with_positive_realized_full_gain=sum(r['realized_full_gain_per_asset']>1e-10 for r in reached),
        candidate_same_stage_as_static=sum(r['methods']['dual_temporal']['stage']==r['methods']['dual_static']['stage'] for r in records),
        candidate_same_stage_as_risk_only=sum(r['methods']['dual_temporal']['stage']==r['methods']['risk_temporal']['stage'] for r in records))
    save(out/'mechanism_diagnostics.json',dict(status='POSTHOC descriptive mechanism analysis; all32 included; no model/threshold edits',
        limits='Realized dependent outcomes, not expected gain, Bayes monotonicity, independent event counts, or a prospective policy',
        summaries=summaries,records=records,gain_blocks_all_rank_states=blocks,loss_increases=increases))
    print(summaries,flush=True)


if __name__=='__main__':main()
