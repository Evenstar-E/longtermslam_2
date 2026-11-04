"""Posthoc failure decomposition; never relabel surface fit as motion ground truth."""
import json,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.score_real_temporal_v1 import read,save

def main():
    out=Path('results/real_temporal_v1');root=Path('data/real_cross_session_v1/prepared_v2')
    scores=read(out/'scores.json');cfg=read(out/'config.json');truth=read(root/'evaluation/identity.json');public=read(root/'public.json')
    fits=[json.loads(s) for s in (out/'candidate_fits.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for row in scores['rows']:
        if not row['persistent_eligible']:continue
        options=[r for r in fits if r['scene']==row['scene'] and r['current']==row['current'] and r['method']==row['method']]
        correct=[r for r in options if truth[row['scene']]['0'][r['history']] in row['allowed_ids']]
        wrong=[r for r in options if truth[row['scene']]['0'][r['history']] not in row['allowed_ids']]
        best=min(correct,key=lambda r:r['cost']) if correct else None
        distractor=min(wrong,key=lambda r:r['cost']) if wrong else None
        chosen=[r for r in options if truth[row['scene']]['0'][r['history']]==row['history_id']]
        if row['equivalence_correct']:reason='correct'
        elif best is None:reason='no_same_class_candidate'
        elif best['cost']>=cfg['null_assignment_cost']:reason='correct_pair_above_shared_acceptance_cost'
        elif distractor and distractor['cost']<best['cost']:reason='wrong_identity_has_lower_individual_cost'
        else:reason='global_assignment_conflict_or_null_preference'
        rows.append(dict(row,reason=reason,correct_pair_cost=None if best is None else best['cost'],
            distractor_cost=None if distractor is None else distractor['cost'],
            chosen_fit=None if not chosen else chosen[0]['stats']))
    summary=[dict(method=m,counts=dict(Counter(r['reason'] for r in rows if r['method']==m)),
        wrong_but_low_surface_cost=sum(r['status']=='wrong_identity' and r['chosen_fit'] is not None and r['chosen_fit']['cost']<.35 for r in rows if r['method']==m)) for m in cfg['methods']]
    save(out/'failure_decomposition.json',{'summary':summary,'rows':rows,'status':'POSTHOC diagnostic; costs depend on this adapter and gates, not a causal theorem about the published full systems'})
    print(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    pooled=[r for r in scores['summary'] if r['scene']=='all']
    x=np.arange(len(pooled));bottom=np.zeros(len(x))
    for field,label,color in [('correct','Correct ID','#009E73'),('persistent_wrong','Wrong ID','#D55E00'),('persistent_unmatched','Unmatched','#999999')]:
        vals=np.array([r[field] for r in pooled]);axs[0].barh(x,vals,left=bottom,label=label,color=color);bottom+=vals
    axs[0].set(yticks=x,yticklabels=[r['method'] for r in pooled],xlabel='Objects (62 persistent observed instances)',title='A. Real identity association');axs[0].legend(fontsize=8)
    for i,r in enumerate(scores['diagnostics']):
        if r['anchor_displacement_span_m'] is not None:axs[1].scatter(i,r['anchor_displacement_span_m'],s=22,c=colors[cfg['spaces'].index(r['scene'])],marker=['o','s','^'][cfg['spaces'].index(r['scene'])])
    axs[1].axhline(.1,color='black',ls='--',lw=1);axs[1].set(xlabel='Strict persistent pair index',ylabel='Anchor estimate disagreement (m)',title='B. Multiple well-fitting estimates');axs[1].set_ylim(bottom=0)
    color=read('results/real_temporal_color_control_v1/scores.json');rr=[r for r in color['summary'] if r['scene']=='all']
    for j,r in enumerate(rr):
        axs[2].bar(j,r['correct'],color='#0072B2');axs[2].text(j,r['correct']+1,str(r['correct']),ha='center')
    axs[2].set(xticks=np.arange(3),xticklabels=['RGB','RGB + position','RGB + surface'],ylim=(0,62),ylabel='Correct ID / 62',title='C. Posthoc ordinary RGB control');axs[2].tick_params(axis='x',rotation=20)
    fig.suptitle('Real Rescan: privileged masks/classes; 3 spaces. No metric object-motion ground truth.',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    # Raw colored reconstructions: no synthetic image generation or recoloring.
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    from scripts.prepare_real_temporal_v1 import read_ply
    for j,s in enumerate(cfg['spaces']):
        for capture in [0,1]:
            p=read_ply(Path('data/real_cross_session_v1/rescan/raw')/s/'gt_segmentation'/f'{s}_{capture}.ply')
            sample=p[::max(1,len(p)//30000)]
            rgb=np.column_stack([sample[n] for n in ['red','green','blue']])/255
            axs[capture,j].scatter(sample['x'],sample['z'],c=rgb,s=.6,rasterized=True)
            axs[capture,j].set(title=f'{s}, capture {capture}',xlabel='x (m)',ylabel='z (m)',aspect='equal')
    fig.suptitle('Acquired real reconstructions, top-view x-z projections, original RGB')
    fig.savefig(out/'real_inputs.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
