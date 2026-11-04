"""Descriptive full-task and complete common-initial-state branch figures."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS=dict(legacy_candidate='Original candidate',legacy_risk='Original task risk',legacy_schur='Original information',
    weighted_candidate='Weighted paired candidate',weighted_risk='Weighted paired risk',
    uniform_candidate='Uniform paired candidate',uniform_risk='Uniform paired risk',
    support_coverage='Missing support coverage',represented_coverage='Represented coverage',
    events_candidate='Event likelihood + candidate',events_risk='Event likelihood + risk',
    constant_events_candidate='Constant events + candidate',constant_events_risk='Constant events + risk')


def main():
    base=Path('results/comparative_selection_v7');a=json.loads((base/'analysis.json').read_text());s=a['summaries']
    methods=list(s);y=np.arange(len(methods));labels=[f"{LABELS[m]} [FP={s[m]['risk']['fp']}]" for m in methods]
    qualified=np.array([s[m]['risk']['qualified_tp'] for m in methods]);bad=np.array([s[m]['risk']['bad_motion_reports'] for m in methods])
    rest=72-np.array([s[m]['risk']['tp'] for m in methods])
    fig,axes=plt.subplots(1,2,figsize=(12,8.5),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax=axes[0];ax.barh(y,qualified,color='#009E73',label='Qualified report')
    ax.barh(y,bad,left=qualified,color='#E69F00',hatch='//',label='Wrong displacement report')
    ax.barh(y,rest,left=qualified+bad,color='#CCCCCC',hatch='..',label='No change report (incl. unknown)')
    for i,v in enumerate(qualified):ax.text(v/2,i,str(v),ha='center',va='center',color='white')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,72);ax.set_xlabel('All 72 true changes')
    ax.set_title('Full budget: history4 / extra current2')
    ax.legend(loc='lower left',bbox_to_anchor=(-.65,-.22),frameon=False,fontsize=8)
    ax=axes[1];loss=[s[m]['risk']['task_loss'] for m in methods];ax.barh(y,loss,color='#0072B2')
    for i,v in enumerate(loss):ax.text(v+.1,i,f'{v:.2f}',va='center',fontsize=8)
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(0,max(loss)*1.2)
    ax.set_xlabel('Total declared loss /384 asset slots');ax.set_title('Lower is better; artificial task costs')
    fig.suptitle('48 reused DEV scenes;192 new policies; all v5/v6 controls retained',fontsize=11)
    for ext in ('png','svg'):fig.savefig(base/f'comparison.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    b=a['branch_validation']['top_action_summaries'];methods=list(b);y=np.arange(len(methods))
    labels=[f"{LABELS[m]} [FP={b[m]['fp_after_one_action']}]" for m in methods]
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
    ax=axes[0];loss=[b[m]['total_taskloss_after_one_action'] for m in methods];ax.barh(y,loss,color='#0072B2')
    for i,v in enumerate(loss):ax.text(v+.12,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,max(loss)*1.2)
    ax.set_title('Actual loss after top predicted initial action');ax.set_xlabel('Total declared loss /384 asset slots')
    ax=axes[1]
    for i,m in enumerate(methods):
        rho=b[m]['mean_within_scene_rank_rho']
        if rho is not None:ax.scatter(rho,i,color='#009E73');ax.text(rho+.025,i,f'{rho:.2f}',va='center')
        else:ax.text(0,i,'undefined',va='center')
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(-1,1);ax.axvline(0,color='#999999',lw=.8)
    ax.set_title('Mean within-scene action rank correlation');ax.set_xlabel('Descriptive Spearman rho; no action-level p-values')
    fig.suptitle('Offline validation only: all22 actions on each common initial H0/C0 state',fontsize=11)
    for ext in ('png','svg'):fig.savefig(base/f'branch_validation.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    # Raw RGB diagnostic: no evaluator masks, recoloring, crop or image synthesis.
    fig,axes=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
    for ax,token,title in zip(axes,('history_00','history_06'),
        ('Known H0 before action','H6 after legal branch action')):
        with np.load(Path('results/comparative_interface_v1/scenes/0000/payloads')/f'{token}.npz') as data:
            ax.imshow(data['rgb'],interpolation='nearest')
        ax.axis('off');ax.set_title(title)
    fig.suptitle('Scene0: predicted H6 detection1% for every asset; actual8/8 detected',fontsize=10)
    fig.savefig(base/'projection_mismatch.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    fields=['method','variant','qualified_tp','tp','fp','bad_motion_reports','known','task_loss','mean_travel_m']
    with (base/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m in s:
            for v,r in s[m].items():w.writerow(dict(method=m,variant=v,**{k:r[k] for k in fields[2:]}))
    files=['comparison.png','comparison.svg','branch_validation.png','branch_validation.svg','projection_mismatch.png','summary.csv']
    manifest=dict(source_sha256=hashlib.sha256((base/'analysis.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        scope='All controls;39 terminal summaries; no filtered changes, independent uncertainty bars or smoothing',
        outputs={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in files})
    (base/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
