"""Evaluator only, called after all four inference predictions were sealed."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.completion_compatible_anchor_v1 import conflict

def main():
    root=Path('results/completion_compatible_anchor_v1');seal=read(root/'prediction_seal.json')
    assert seal['complete']
    for path,value in seal['sha256'].items():assert sha(root/path)==value
    rows=[];source_checks=[]
    for run in read(root/'runs.json'):
        dest=root/run['scene']
        with (dest/'inference.pkl').open('rb') as f:p=pickle.load(f)
        with (dest/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        masks=np.load(dest/'feasible_masks.npz');records=read(dest/'records.json');pred=read(dest/'predictions.json')
        for i,(h,bb,a,rec) in enumerate(zip(p['history'],p['branches'],p['anchors'],records)):
            correct=[]
            for t in [0,1]:
                correct.append(next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]))
            groups=[h,bb[0][correct[0]],bb[1][correct[1]]];S=truth['occupancy'][i]
            invalid_free=[(S&g['free']).bit_count() for g in groups]
            invalid_bands=[sum(not(b&S) for b in g['bands']) for g in groups]
            source_checks.append({'scene':run['scene'],'asset':i,'forbidden_true_occupied':invalid_free,'hit_bands_without_true_occupancy':invalid_bands,
                                  'true_pairwise_retained':bool(masks[f'pair_{i}'][tuple(correct)]),'true_full_retained':bool(masks[f'full_{i}'][tuple(correct)])})
            for method,result in pred.items():
                selection=result['selection'];chosen=next((v for v in selection if v[0]==i),None) if selection else None
                row={'scene':run['scene'],'asset':i,'method':method,'resolved':chosen is not None,'id_both':False,'anchor_error_m':None,'joint_success_5cm':False}
                if chosen:
                    _,k,l=chosen;one,two=bb[0][k],bb[1][l]
                    d=two['T'][:3,:3]@a+two['T'][:3,3]-one['T'][:3,:3]@a-one['T'][:3,3]
                    id_ok=one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]
                    error=float(np.linalg.norm(d-truth['anchor_displacement'][i]));row.update(id_both=bool(id_ok),anchor_error_m=error,joint_success_5cm=bool(id_ok and error<=.05))
                rows.append(row)
            for method in ['pairwise','full']:
                env=rec[f'{method}_envelope'];d=truth['anchor_displacement'][i]
                contained=not env['empty'] and bool(np.all(d>=np.array(env['lower'])-1e-10) and np.all(d<=np.array(env['upper'])+1e-10))
                estimate=(np.array(env['lower'])+np.array(env['upper']))/2 if not env['empty'] else None
                error=float(np.linalg.norm(estimate-d)) if estimate is not None else None
                rows.append({'scene':run['scene'],'asset':i,'method':method+'_envelope','empty':env['empty'],
                             'contains_true_displacement':contained,'report':env['report'],'radius_m':env['radius_m'],
                             'report_error_m':error if env['report'] else None,
                             'wrong_report_5cm':bool(env['report'] and error>.05)})
    save(root/'scores.json',rows);save(root/'observation_soundness.json',source_checks)
    summary={}
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method]
        if method.endswith('envelope'):
            summary[method]={'n':len(rr),'reports':sum(r['report'] for r in rr),'wrong_reports':sum(r['wrong_report_5cm'] for r in rr),
                             'contains_truth':sum(r['contains_true_displacement'] for r in rr),'empty':sum(r['empty'] for r in rr)}
        else:summary[method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'both_epoch_ids':sum(r['id_both'] for r in rr),
                             'identity_and_anchor_5cm':sum(r['joint_success_5cm'] for r in rr),
                             'median_anchor_error_m':float(np.median([r['anchor_error_m'] for r in rr if r['resolved']])) if any(r['resolved'] for r in rr) else None}
    summary['soundness']={'n':len(source_checks),'true_pairs_retained':sum(r['true_full_retained'] for r in source_checks),
                          'true_occupied_forbidden':sum(sum(r['forbidden_true_occupied']) for r in source_checks),
                          'hit_bands_missing_truth':sum(sum(r['hit_bands_without_true_occupancy']) for r in source_checks)}
    save(root/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
