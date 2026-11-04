"""Annotation evaluation after the complete inference seal, including abstention."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    cfg=read('configs/material_observation_consistency_v1.json');out=Path(cfg['output_root']);root=Path(cfg['public_root']);poolroot=Path(cfg['pool_root'])
    for name,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/name)==h
    for field in ['sources','inputs']:
        for name,h in read(out/'manifest.json')[field].items():assert sha(name)==h
    rows=[];diagnosis=[]
    for folder in sorted(out.glob('pair_*')):
        p=read(folder/'predictions.json');pool=read(poolroot/folder.name/'predictions.json');pair=p['pair']
        h=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);c=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4)
        GT=c@np.linalg.inv(h);anchor=np.array(pool['anchor']);target=GT[:3,:3]@anchor+GT[:3,3]
        errors=[float(np.linalg.norm(np.array(e['T_current_from_history'])[:3,:3]@anchor+np.array(e['T_current_from_history'])[:3,3]-target)) for e in pool['pool']]
        for method,i in p['choices'].items():
            rows.append(dict(pair=folder.name,method=method,accepted=i is not None,pose_index=i,anchor_error_m=errors[i] if i is not None else None,support=p['rows'][i]['support'] if i is not None else None))
        esc=pool['escape_indices'];correct=[i for i,e in enumerate(errors) if e<=.01]
        d=dict(pair=folder.name,poses=len(errors),oracle_anchor_error_m=min(errors),within_1cm_candidates=len(correct),escape_candidates=len(esc),
               rejected={g:sum(not r['gates'][g] for r in p['rows']) for g in ['raw','stable']},
               escape_rejected={g:sum(not p['rows'][i]['gates'][g] for i in esc) for g in ['raw','stable']},methods={})
        for m in ['nn','visibility']:
            i=p['choices'][m];best=min(correct,key=lambda j:p['rows'][j]['scores'][m]) if correct else None
            d['methods'][m]=dict(selected=i,selected_error_m=errors[i],best_correct=best,best_correct_score=p['rows'][best]['scores'][m] if best is not None else None,
                                  selected_score=p['rows'][i]['scores'][m],best_correct_survives={g:p['rows'][best]['gates'][g] for g in ['raw','stable']} if best is not None else None)
        diagnosis.append(d)
    summary=[]
    for m in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==m];e=np.array([r['anchor_error_m'] for r in rr if r['accepted']])
        summary.append(dict(method=m,total=len(rr),accepted=len(e),abstained=len(rr)-len(e),within_1cm=int((e<=.01).sum()),within_10cm=int((e<=.1).sum()),accepted_wrong_1cm=int((e>.01).sum()),median_mm=float(np.median(e)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None))
    save(out/'scores.json',dict(summary=summary,rows=rows,GT_read_only_after_seal=True,scope=cfg['scope']))
    save(out/'pool_diagnosis.json',dict(pairs=diagnosis,GT_assisted=True,predictions_changed=False))
    print(summary,flush=True);print('rejected',{g:sum(d['rejected'][g] for d in diagnosis) for g in ['raw','stable']},flush=True)


if __name__=='__main__':main()
