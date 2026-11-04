"""Posthoc oracle ablations locate uncertainty bottlenecks, not deployed methods."""
import json
from pathlib import Path
import numpy as np
from scripts.map_evidence_v4b import Model


def labels(model,cfg):
    base=model.solve()
    if base.status!=0:return [-1]*len(model.q)
    result=[];tau=cfg['motion_axis_threshold_m']
    for j in model.q:
        possible=False;ok=True
        for a in range(3):
            for sign in (1,-1):
                obj=np.zeros(model.size);obj[model.d0+3*j+a]=sign;r=model.solve(obj)
                if r.status!=0:ok=False
                elif abs(r.x[model.d0+3*j+a])>tau+1e-5:possible=True
        lb=model.lower.copy();ub=model.upper.copy();lb[model.d0+3*j:model.d0+3*j+3]=-tau;ub[model.d0+3*j:model.d0+3*j+3]=tau;r=model.solve(np.zeros(model.size),(lb,ub))
        result.append(1 if ok and r.status==2 else (0 if ok and r.status==0 and not possible else -1))
    return result


def main():
    out=Path('results/map_evidence_v4b');cfg=json.loads((out/'frozen_config.json').read_text());summary={k:dict(known=0,visible=0,tp=0,fp=0,changed=0) for k in ('oracle_calibration','oracle_reference','oracle_both')};records=[]
    for sid in range(64,128):
        o=json.loads((out/'cases'/f'{sid:04d}_input.json').read_text());t=json.loads((out/'cases'/f'{sid:04d}_truth.json').read_text());truth=np.array(t['changed']);correct=np.array(o['assignment'])==t['assignment']
        for method in summary:
            m=Model(o,cfg)
            if method in ('oracle_calibration','oracle_both'):
                u=np.r_[np.array(t['realized_history_nuisance']).reshape(-1),t['realized_query_nuisance']];m.lower[m.u0:m.z0]=u;m.upper[m.u0:m.z0]=u
            if method in ('oracle_reference','oracle_both'):
                box=np.array(t['reference_boxes']).transpose(0,2,1).reshape(-1);m.lower[:m.d0]=box;m.upper[:m.d0]=box
            lab=np.array(labels(m,cfg));s=summary[method];s['known']+=int((lab>=0).sum());s['visible']+=len(lab);s['tp']+=int(((lab==1)&truth&correct).sum());s['fp']+=int(((lab==1)&(~truth|~correct)).sum());s['changed']+=int(truth.sum());records.append(dict(scene=sid,method=method,labels=lab.tolist()))
    for s in summary.values():s['coverage']=s['known']/s['visible'];s['recall']=s['tp']/s['changed']
    result=dict(scope='Posthoc fresh-set oracle cause isolation only; truth never given to primary inference; not executable performance claims',summary=summary,records=records)
    (out/'oracle_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
