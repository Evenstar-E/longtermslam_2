"""All retained controls, full terminal CSV, descriptive selected-step diagnostics."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path('results/comparative_selection_v6');a=json.loads((base/'analysis.json').read_text())
    methods=list(a['summaries']);s=a['summaries'];y=np.arange(len(methods))
    names=['Original candidate','Original task risk','Original information','Weighted paired candidate',
        'Weighted paired task risk','Uniform paired candidate','Uniform paired task risk',
        'Missing support coverage','Represented coverage']
    labels=[f"{name} [FP={s[m]['risk']['fp']}]" for name,m in zip(names,methods)]
    qualified=np.array([s[m]['risk']['qualified_tp'] for m in methods])
    bad=np.array([s[m]['risk']['bad_motion_reports'] for m in methods])
    no_report=72-np.array([s[m]['risk']['tp'] for m in methods])
    fig,axes=plt.subplots(1,2,figsize=(12,6.3),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax=axes[0];ax.barh(y,qualified,color='#009E73',label='Qualified report')
    ax.barh(y,bad,left=qualified,color='#E69F00',hatch='//',label='Wrong displacement report')
    ax.barh(y,no_report,left=qualified+bad,color='#CCCCCC',hatch='..',label='No change report (incl. unknown)')
    for i,v in enumerate(qualified):ax.text(v/2,i,str(v),ha='center',va='center',color='white')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,72);ax.set_xlabel('All 72 true changes')
    ax.set_title('Shared 2048-point terminal; all false changes labeled')
    ax.legend(loc='lower left',bbox_to_anchor=(-.65,-.27),frameon=False,fontsize=8)
    ax=axes[1];loss=[s[m]['risk']['task_loss'] for m in methods]
    ax.barh(y,loss,color='#0072B2')
    for i,v in enumerate(loss):ax.text(v+.08,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_yticks(y,[]);ax.invert_yaxis();ax.set_xlim(0,max(loss)*1.2)
    ax.set_xlabel('Total declared task loss /384 asset slots');ax.set_title('Lower is better; artificial task costs')
    fig.suptitle('48 reused DEV scenes: 96 new policies +96 diagnostic replays; v5 controls retained',fontsize=11)
    for ext in ('png','svg'):fig.savefig(base/f'comparison.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    fields=['method','variant','qualified_tp','tp','fp','bad_motion_reports','known','task_loss','mean_travel_m']
    with (base/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m in methods:
            for v,r in s[m].items():writer.writerow(dict(method=m,variant=v,**{k:r[k] for k in fields[2:]}))
    manifest=dict(source_sha256=hashlib.sha256((base/'analysis.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        scope='All changed slots retained; no smoothing/selection/error bars; paired scene intervals in report; point/two_sigma FP in CSV',
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (base/'comparison.png',base/'comparison.svg',base/'summary.csv')})
    (base/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
