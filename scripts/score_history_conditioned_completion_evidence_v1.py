"""Postseal evaluator; no new fit or score/threshold modification."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha

def main(output='results/history_conditioned_completion_evidence_v1'):
    root=Path(output);cfg=read(root/'config.json');base=Path(cfg['input_root'])
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    physical=read(base/'material_anchor_diagnosis.json')['sources'];physical={(r['scene'],r['asset']):np.array(r['physical_material_anchor']) for r in physical}
    rows=[];sampling=[]
    for run in read(root/'runs.json'):
        scene=run['scene'];folder=base/scene
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        with (folder/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(root/scene/'predictions.json');scores=np.load(root/scene/'scores.npz')
        for method,result in pred.items():
            selection={i:(k,l) for i,k,l in result['selection']} if result['selection'] is not None else {}
            for i,(bb,a) in enumerate(zip(p['branches'],p['anchors'])):
                row={'scene':scene,'asset':i,'method':method,'resolved':i in selection,'both_ids_correct':False,
                     'joint_reference_5cm':False,'joint_physical_5cm':False,'physical_error_m':None}
                if i in selection:
                    k,l=selection[i];one,two=bb[0][k],bb[1][l]
                    estimate=two['T'][:3,:3]@a+two['T'][:3,3]-one['T'][:3,:3]@a-one['T'][:3,3]
                    ma=physical[scene,i]
                    target=truth['poses'][1,i,:3,:3]@ma+truth['poses'][1,i,:3,3]-truth['poses'][0,i,:3,:3]@ma-truth['poses'][0,i,:3,3]
                    ids=one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]
                    err=float(np.linalg.norm(estimate-target));reference_error=float(np.linalg.norm(estimate-truth['anchor_displacement'][i]))
                    correct=tuple(next(j for j,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1])
                    array=scores[f'{method}_{i}']
                    row.update(both_ids_correct=bool(ids),joint_reference_5cm=bool(ids and reference_error<=.05),joint_physical_5cm=bool(ids and err<=.05),
                               physical_error_m=err,selected_k1=k,selected_k2=l,true_k1=correct[0],true_k2=correct[1],
                               selected_minus_true_logscore=float(array[k,l]-array[correct]) if np.isfinite(array[correct]) else None)
                rows.append(row)
        sampling.extend(read(root/scene/'sampling_errors.json'))
    summary={};conditions={}
    for method in cfg['methods']:
        rr=[r for r in rows if r['method']==method]
        summary[method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'both_ids':sum(r['both_ids_correct'] for r in rr),
                         'joint_reference_5cm':sum(r['joint_reference_5cm'] for r in rr),'joint_physical_5cm':sum(r['joint_physical_5cm'] for r in rr)}
        conditions[method]={s:sum(r['joint_physical_5cm'] for r in rr if r['scene']==s) for s in sorted({r['scene'] for r in rr})}
    sampling_summary={}
    for particles in cfg['particles']:
        rr=[r for r in sampling if r['particles']==particles];error=np.array([abs(r['log_error']) for r in rr if r['log_error'] is not None])
        sampling_summary[str(particles)]={'events':len(rr),'zero_estimates':sum(r['zero_estimate'] for r in rr),
                                           'median_abs_logerror':float(np.median(error)),'p95_abs_logerror':float(np.quantile(error,.95)),
                                           'max_abs_logerror':float(error.max())}
    save(root/'scores.json',rows);save(root/'summary.json',{'methods':summary,'conditions':conditions,'sampling':sampling_summary,
                                                           'old_full_constraint_baseline_joint_physical_5cm':60})
    print({'methods':summary,'conditions':conditions,'sampling':sampling_summary},flush=True)

if __name__=='__main__':main()
