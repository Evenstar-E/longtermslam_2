"""All-method figures and complete post-hoc mechanism diagnostics, no retuning."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths
from inspection_decisions.graph_dataset import digest
from inspection_decisions.optional_stopping_v1 import decision


LABELS=['Optional CF temporal (candidate)','Ordinary finite horizon','Optional CF static',
        'One-step value','Full-tail value','Optional without risk cap','Previous dual temporal',
        'Previous dual static','Temporal risk only','Declared risk / rank','Fixed stage / rank',
        'Original STOP / rank','Full budget / rank','Declared risk / coverage','Full budget / coverage']


def main():
    out=Path('results/optional_stopping_v1');a=read(out/'analysis.json');cfg=read('configs/optional_stopping_v1.json')
    if (out/'mechanism_diagnostics.json').exists():raise FileExistsError('report already exists')
    bundle=read(out/'models.json');methods=cfg['methods'];paths=load_paths(out,'evaluation');records=[];increases=[]
    by={(r['scene'],r['method']):r for r in a['rows']}
    for p in paths:
        if p['policy']!='rank':continue
        sid=p['scene'];states=p['observed']['states'];metrics=p['labels_data']['metrics'];L=[m['task_loss'] for m in metrics]
        previous=None;all_decisions=[]
        for state in states:
            t=state['stage'];values={}
            for method in methods[:6]:
                stop,detail=decision(method,state,previous,states[0]['prediction'],bundle,bundle['settings'][method])
                values[method]=dict(stop=stop,**detail)
            all_decisions.append(dict(stage=t,values=values,actual_loss=L[t]))
            if t<5 and L[t+1]>L[t]+1e-10:increases.append(dict(scene=sid,stage=t,before=L[t],after=L[t+1],action=state['selected_option'][0]))
            previous=state['prediction']
        records.append(dict(scene=sid,shift=p['shift'],loss_sequence=L,actions=[s['selected_option'][0] for s in states[:-1]],
            methods={m:dict(stage=by[sid,m]['stage'],loss=by[sid,m]['metric']['task_loss'],qualified=by[sid,m]['metric']['qualified_tp'],
                fp=by[sid,m]['metric']['fp'],bad_motion=by[sid,m]['metric']['bad_motion_reports'],unknown=by[sid,m]['unknown']) for m in methods},
            all_state_value_predictions=all_decisions))
    summaries={}
    for other in methods[1:]:
        c='optional_cf_temporal'
        summaries[other]=dict(same_stop=sum(r['methods'][c]['stage']==r['methods'][other]['stage'] for r in records),
            better_loss=sum(r['methods'][c]['loss']<r['methods'][other]['loss']-1e-10 for r in records),
            worse_loss=sum(r['methods'][c]['loss']>r['methods'][other]['loss']+1e-10 for r in records),
            equal_loss=sum(abs(r['methods'][c]['loss']-r['methods'][other]['loss'])<=1e-10 for r in records))
    diagnostics=dict(status='POSTHOC descriptive; all32families; no model or threshold changes',
        limits='Dependent realized changes, not Bayes risk monotonicity evidence or optimal future truth labels',
        rank_step_loss_increases=len(increases),rank_steps=160,comparisons=summaries,records=records,loss_increases=increases)
    save(out/'mechanism_diagnostics.json',diagnostics)
    table=a['summaries']['all'];y=np.arange(len(methods));cols=['#0072B2']+['#777777']*(len(methods)-1)
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['task_loss','mean_acquisitions','mean_return_travel_m'],
            ['Total task loss (lower is better)','Mean additional acquisitions','Mean travel incl. return (m)']):
        vals=[table[m][key] for m in methods];ax.barh(y,vals,color=cols);ax.set_xlim(0,max(vals)*1.2+.01)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(y,LABELS);axes[0].invert_yaxis()
    fig.suptitle('Fresh 32 families: 48 changes, 208 unchanged assets\n64 actual full acquisition paths; 480 causal STOP outcomes, not independent reruns',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    baselines=methods[1:];y=np.arange(len(baselines));fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['task_loss','acquisitions','travel_m'],
                           ['Mean report-loss difference','Mean acquisition difference','Mean travel difference (m)']):
        vals=[a['contrasts']['all'][m][key] for m in baselines];mean=np.array([v['mean'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
        ax.errorbar(mean,y,xerr=np.maximum(np.array([mean-ci[:,0],ci[:,1]-mean]),0),fmt='o',capsize=3,color='#0072B2')
        ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=11)
    axes[0].set_yticks(y,LABELS[1:]);axes[0].invert_yaxis()
    fig.suptitle('Candidate minus comparator (negative favors candidate on these axes)\n5000 paired family bootstrap; stratified conditions, exploratory unadjusted95%',fontsize=11)
    fig.savefig(out/'paired_differences.png',dpi=180);fig.savefig(out/'paired_differences.svg');plt.close(fig)
    save(out/'figure_manifest.json',dict(analysis_sha256=digest(out/'analysis.json'),script_sha256=digest(__file__),
        numpy=np.__version__,matplotlib=matplotlib.__version__,purpose='Research review, no journal-specific compliance claim',
        transformations='All15methods; observed aggregate bars fromzero; paired family mean differences with percentile95%CI',
        exclusions=[],files={p.name:digest(p) for p in out.glob('*.png')}))
    print('Mechanism loss increases',len(increases),'/',160,flush=True)
    for m in ['optional_in_temporal','full_tail_temporal','original_rank']:print(m,summaries[m],flush=True)


if __name__=='__main__':main()
