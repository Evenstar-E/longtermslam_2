"""POST-HOC attribution only; no refitting or development tuning."""
import copy
from pathlib import Path
import numpy as np
from inspection_decisions.value_pilot_v1 import predict_value,choose
from scripts.run_evidence_value_pilot_v1 import read,save,outcome_arrays,summary,SPLITS


def main():
    out=Path('results/evidence_value_pilot_v1');idx=read(out/'input_index.json');cfg=read('configs/evidence_value_pilot_v1.json')
    X=np.load(out/'features.npz')['X'];dev=np.array(idx['splits'])==SPLITS[2]
    m=read(out/'model_structured_rbf.json')
    for part in ('single','interaction'):
        for key in ('mean','scale','Z','coef'):m[part][key]=np.array(m[part][key])
    _,train_gain=outcome_arrays(read(out/f'{SPLITS[0]}_outcomes.json'))
    pairs=m['pair_indices'];a=[a for a,b in m['pair_members']];b=[b for a,b in m['pair_members']]
    constant=float((train_gain[:,pairs]-train_gain[:,a]-train_gain[:,b]).mean())
    no=copy.deepcopy(m);del no['interaction'];const=copy.deepcopy(no);const['constant_interaction']=constant
    data=read(out/f'{SPLITS[2]}_outcomes.json');counts=np.array([len(o) for o in idx['options']]);price=cfg['primary_acquisition_price']
    results={}
    for name,model in [('frozen_structured',m),('same_single_no_interaction',no),('same_single_train_mean_interaction',const)]:
        choice=choose(predict_value(model,X[dev]),counts,price)
        results[name]=dict(summary=summary(data,choice,counts,price),choices=[idx['option_names'][i] for i in choice])
    save(out/'posthoc_attribution.json',dict(status='POST_HOC after primary evaluation; not replacement primary contrast',
        training_mean_interaction=constant,results=results,
        purpose='Primary independently calibrated models have different alpha; hold single model fixed to isolate residual influence',
        limits='No new fit, no tuning, no confirmatory claim'))
    print({k:v['summary']['task_loss'] for k,v in results.items()})


if __name__=='__main__':main()
