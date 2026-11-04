"""Descriptive counts only; not independent uncertainty or journal formatting."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path('results/comparative_selection_v5');analysis=json.loads((base/'analysis.json').read_text())
    methods=list(analysis['summaries'])
    labels=['Old forecast + candidate','Old forecast + task risk','Old forecast + information',
            'Reciprocal + candidate','Reciprocal + task risk','Missing support coverage','Represented coverage']
    summaries=analysis['summaries'];y=np.arange(len(methods))
    labels=[f"{label} [FP={summaries[m]['risk']['fp']}]" for label,m in zip(labels,methods)]
    qualified=np.array([summaries[m]['risk']['qualified_tp'] for m in methods])
    bad=np.array([summaries[m]['risk']['bad_motion_reports'] for m in methods])
    unreported=72-np.array([summaries[m]['risk']['tp'] for m in methods])
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
    ax=axes[0]
    ax.barh(y,qualified,color='#009E73',label='Qualified report')
    ax.barh(y,bad,left=qualified,color='#E69F00',hatch='//',label='Wrong displacement report')
    ax.barh(y,unreported,left=qualified+bad,color='#CCCCCC',hatch='..',label='No change report (incl. unknown)')
    for i,v in enumerate(qualified):ax.text(v/2,i,str(v),ha='center',va='center',color='white')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,72)
    ax.set_xlabel('All 72 true changes');ax.set_title('Common 2048-point terminal integration')
    ax.legend(loc='lower left',bbox_to_anchor=(-.65,-.36),ncol=1,frameon=False,fontsize=8)
    ax=axes[1]
    point=np.array([summaries[m]['point']['qualified_tp'] for m in methods])
    ax.scatter(point,y,marker='x',color='#0072B2',s=45,label='Direct point rule')
    ax.scatter(qualified,y,marker='o',facecolors='none',edgecolors='#009E73',s=55,label='Common task terminal')
    for i,(a,b) in enumerate(zip(point,qualified)):ax.plot([a,b],[i,i],color='#999999',zorder=0)
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(0,72)
    ax.set_xlabel('Qualified reports /72');ax.set_title('Point-rule control retained')
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.36),frameon=False,fontsize=8)
    fig.suptitle('Development only: 48 reused scenes, 336 actual policy executions',fontsize=12)
    fig.savefig(base/'comparison.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    fig.savefig(base/'comparison.svg',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    fields=['method','variant','qualified_tp','tp','fp','bad_motion_reports','known','task_loss','mean_travel_m']
    with (base/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m in methods:
            for v,s in summaries[m].items():writer.writerow(dict(method=m,variant=v,**{k:s[k] for k in fields[2:]}))
    manifest=dict(source='analysis.json',source_sha256=hashlib.sha256((base/'analysis.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        scope='Descriptive all72 true changes, no smoothing/filtering/error bars; all changed slots retained',
        false_changes={m:summaries[m]['risk']['fp'] for m in methods},
        outputs={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (base/'comparison.png',base/'comparison.svg',base/'summary.csv')})
    (base/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
