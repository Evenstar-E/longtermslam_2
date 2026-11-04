"""All prespecified methods and paired contrasts; no selective plotting."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspection_decisions.graph_dataset import digest
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    out=Path('results/report_risk_stopping_v1');data=read(out/'analysis.json')
    methods=read('configs/report_risk_stopping_v1.json')['methods'];y=np.arange(len(methods))
    labels=['Dual temporal (candidate)','Dual static','Temporal risk only','Declared risk / rank',
            'Fixed stage / rank','Original STOP / rank','Full budget / rank',
            'Declared risk / coverage','Full budget / coverage']
    fig,axes=plt.subplots(1,3,figsize=(14,6),sharey=True,layout='constrained')
    cols=['#0072B2' if m=='dual_temporal' else '#777777' for m in methods]
    table=data['summaries']['all']
    for ax,key,title in zip(axes,['task_loss','mean_acquisitions','mean_return_travel_m'],
                           ['Total task loss (lower is better)','Mean additional acquisitions','Mean travel incl. return (m)']):
        vals=[table[m][key] for m in methods];ax.barh(y,vals,color=cols)
        ax.set_title(title,fontsize=11);ax.set_xlim(0,max(vals)*1.22+.01);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.025,i,f'{v:.2f}',va='center',fontsize=9)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
    fig.suptitle('Fresh evaluation: 32 families, 48 changes, 208 unchanged assets\n64 shared full paths; each stopping rule consumes only its causal prefix',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    baselines=list(data['contrasts']['all']);y=np.arange(len(baselines))
    fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['task_loss','acquisitions','travel_m'],
                           ['Mean task-loss difference','Mean acquisition difference','Mean travel difference (m)']):
        vals=[data['contrasts']['all'][b][key] for b in baselines]
        means=np.array([v['mean'] for v in vals]);bounds=np.array([v['ci95'] for v in vals])
        ax.errorbar(means,y,xerr=np.maximum(np.array([means-bounds[:,0],bounds[:,1]-means]),0),
                    fmt='o',color='#0072B2',capsize=4)
        ax.axvline(0,color='black',lw=1);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,[labels[methods.index(b)] for b in baselines]);axes[0].invert_yaxis()
    fig.suptitle('Candidate minus comparator: negative favors candidate for these metrics\n95% paired family bootstrap, 5000 replicates; condition-stratified, exploratory, unadjusted',fontsize=11)
    fig.savefig(out/'paired_differences.png',dpi=180);fig.savefig(out/'paired_differences.svg');plt.close(fig)
    save(out/'figure_manifest.json',dict(analysis_sha256=digest(out/'analysis.json'),script_sha256=digest(__file__),
        numpy=np.__version__,matplotlib=matplotlib.__version__,purpose='Research review; no journal-specific compliance claim',
        exclusions=[],comparison='Observed aggregates, all nine methods, common zero baselines; no independent-state error bars',
        paired='5000 family bootstrap unadjusted95%, prespecified contrasts, all32 stratified control/stress',
        files={p.name:digest(p) for p in out.glob('*.png')}))


if __name__=='__main__':main()
