"""Posthoc minimum-witness cardinality diagnosis, no changed predictions."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save
from inspection_decisions.history_conditioned_completion_evidence_v1 import event,CounterCompiler

def description(groups,compiler):
    s=event(groups);count=compiler.count(s['clauses']);j=int(np.flatnonzero(count>0)[0])
    return {'minimum_occupied':s['unit_count']+j,'forced_units':s['unit_count'],
            'minimum_residual':j,'log_minimum_multiplicity':float(np.log(count[j])),'active_variables':s['active_count']}

def main():
    root=Path('results/history_conditioned_completion_evidence_v1');base=Path(read(root/'config.json')['input_root'])
    scores=[r for r in read(root/'scores.json') if r['method']=='beta_shared_exact']
    compiler=CounterCompiler();inputs={};rows=[]
    for r in scores:
        scene=r['scene']
        if scene not in inputs:
            with (base/scene/'inference.pkl').open('rb') as f:inputs[scene]=pickle.load(f)
        p=inputs[scene];i=r['asset'];h=p['history'][i];bb=p['branches'][i]
        a=description([h,bb[0][r['selected_k1']],bb[1][r['selected_k2']]],compiler)
        b=description([h,bb[0][r['true_k1']],bb[1][r['true_k2']]],compiler)
        rows.append({'scene':scene,'asset':i,'wrong_identity':not r['both_ids_correct'],
                     'logscore_gap':r['selected_minus_true_logscore'],'selected':a,'true':b,
                     'minimum_occupied_delta':a['minimum_occupied']-b['minimum_occupied']})
    wrong=[r for r in rows if r['wrong_identity'] and r['logscore_gap']>1e-8]
    summary={'wrong_better_cases':len(wrong),'fewer_minimum_occupied':sum(r['minimum_occupied_delta']<0 for r in wrong),
             'equal_minimum_occupied':sum(r['minimum_occupied_delta']==0 for r in wrong),
             'more_minimum_occupied':sum(r['minimum_occupied_delta']>0 for r in wrong),
             'median_minimum_occupied_delta':float(np.median([r['minimum_occupied_delta'] for r in wrong]))}
    save(root/'minimum_witness_diagnosis.json',{'GT_assisted':True,'changes_predictions':False,'summary':summary,'rows':rows})
    print(summary)

if __name__=='__main__':main()
