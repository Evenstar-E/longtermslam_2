"""All17 methods/51 terminals; descriptive figures, no selective winner plot."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.plot_comparative_selection_v7 import LABELS as OLD_LABELS

LABELS=dict(OLD_LABELS,shared_candidate='Shared rays + candidate',shared_risk='Shared rays + risk',
            factorized_candidate='Factorized rays + candidate',factorized_risk='Factorized rays + risk')


def main():
    base=Path('results/comparative_selection_v8');a=json.loads((base/'analysis.json').read_text());s=a['summaries']
    methods=list(s);y=np.arange(len(methods));q=np.array([s[m]['risk']['qualified_tp'] for m in methods])
    bad=np.array([s[m]['risk']['bad_motion_reports'] for m in methods]);rest=72-np.array([s[m]['risk']['tp'] for m in methods])
    fig,axes=plt.subplots(1,2,figsize=(13,10),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained')
    ax=axes[0];ax.barh(y,q,color='#009E73',label='Qualified report')
    ax.barh(y,bad,left=q,color='#E69F00',hatch='//',label='Wrong motion report')
    ax.barh(y,rest,left=q+bad,color='#CCCCCC',hatch='..',label='No change report incl. unknown')
    for i,v in enumerate(q):ax.text(v/2,i,str(v),ha='center',va='center',color='white')
    ax.set_yticks(y,[f"{LABELS[m]} [FP={s[m]['risk']['fp']}]" for m in methods]);ax.invert_yaxis()
    ax.set_xlim(0,72);ax.set_xlabel('All72 true changed assets');ax.set_title('Full budget history4 / extra current2')
    ax.legend(loc='upper left',bbox_to_anchor=(-.8,-.08),frameon=False,fontsize=8)
    loss=[s[m]['risk']['task_loss'] for m in methods];ax=axes[1];ax.barh(y,loss,color='#0072B2')
    for i,v in enumerate(loss):ax.text(v+.12,i,f'{v:.2f}',va='center',fontsize=8)
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(0,max(loss)*1.2);ax.set_xlabel('Declared task loss over384 asset slots')
    ax.set_title('Lower is better; artificial costs')
    fig.suptitle('48 reused DEV scenes;192 new ray policies; all13 older methods retained',fontsize=11)
    for ext in ('png','svg'):fig.savefig(base/f'comparison.{ext}',dpi=170,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    old=json.loads(Path('results/comparative_selection_v7/analysis.json').read_text())['branch_validation']['top_action_summaries']
    b=dict(a['branch_summary'])
    for method in ('events_candidate','uniform_risk','support_coverage'):
        r=old[method];b[method]=dict(qualified=r['qualified_after_one_action'],fp=r['fp_after_one_action'],
            task_loss=r['total_taskloss_after_one_action'],mean_rho=r['mean_within_scene_rank_rho'])
    methods=list(b);y=np.arange(len(methods));fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ax=axes[0];loss=[b[m]['task_loss'] for m in methods];ax.barh(y,loss,color='#0072B2')
    for i,v in enumerate(loss):ax.text(v+.12,i,f'{v:.2f}',va='center')
    ax.set_yticks(y,[f"{LABELS[m]} [q={b[m]['qualified']},FP={b[m]['fp']}]" for m in methods]);ax.invert_yaxis()
    ax.set_xlim(0,max(loss)*1.2);ax.set_xlabel('Actual task loss after selected first action')
    ax=axes[1]
    for i,m in enumerate(methods):
        rho=b[m]['mean_rho']
        if rho is not None:ax.scatter(rho,i,color='#009E73');ax.text(rho+.03,i,f'{rho:.3f}',va='center')
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(-1,1);ax.axvline(0,color='#999999',lw=.8)
    ax.set_xlabel('Descriptive mean within-scene Spearman rho')
    fig.suptitle('All22 initial actions /48 common states; same cached afterfits, new ray observations',fontsize=10)
    for ext in ('png','svg'):fig.savefig(base/f'branch_validation.{ext}',dpi=170,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    c=a['calibration'];fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    axes[0].bar(['Shared','Factorized','Uniform'],[c['shared_nll'],c['factorized_nll'],c['uniform_nll']],color=['#009E73','#0072B2','#999999'])
    axes[0].set_ylabel('Mean joint categorical depth NLL');axes[0].set_ylim(bottom=0)
    axes[1].bar(['Ray marginal','Uniform'],[c['depth_brier'],c['uniform_depth_brier']],color=['#009E73','#999999'])
    axes[1].set_ylabel('Multiclass depth Brier');axes[1].set_ylim(0,1.05)
    fig.suptitle('Same depth observable only; descriptive scores, not task improvement or calibration proof',fontsize=9)
    for ext in ('png','svg'):fig.savefig(base/f'depth_prediction.{ext}',dpi=170,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    files=['comparison.png','comparison.svg','branch_validation.png','branch_validation.svg','depth_prediction.png','depth_prediction.svg','summary.csv']
    manifest=dict(source_sha256=hashlib.sha256((base/'analysis.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in files},
        scope='All17methods51terminalsCSV; all negatives kept; zero axes; descriptive counts, no independent uncertainty bars')
    (base/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
