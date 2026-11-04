"""All fixed-prefix and threshold rules, no selected best threshold."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    base=Path('results/inspection_stopping_audit_v1b');a=json.loads((base/'analysis.json').read_text())
    colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    names=['Uniform task risk','Weighted candidate','Weighted task risk','Uniform candidate']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for (method,rules),color,name in zip(a['summaries'].items(),colors,names):
        fixed=[rules[f'fixed_{i}'] for i in range(6)];threshold=[r for k,r in rules.items() if k.startswith('risk_')]
        for ax,key in zip(axes,['qualified_tp','fp']):
            ax.plot([r['mean_travel_m'] for r in fixed],[r[key] for r in fixed],'-o',color=color,label=name,markersize=4)
            ax.scatter([r['mean_travel_m'] for r in threshold],[r[key] for r in threshold],marker='D',s=45,edgecolor=color,facecolor='none',zorder=4)
    for ax in axes:
        ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.set_xlabel('Mean total travel including immediate return (m)')
        ax.grid(alpha=.18)
    axes[0].set_ylim(0,72);axes[0].set_ylabel('Qualified reports / all72truechanges (higher better)')
    axes[1].set_ylabel('False change reports / all312negatives (lower better)')
    axes[0].legend(loc='lower right',fontsize=8)
    axes[1].legend(handles=[Line2D([0],[0],color='#777777',marker='o',label='Fixed prefixes0..5'),
        Line2D([0],[0],color='#777777',marker='D',linestyle='',markerfacecolor='none',label='All6 declared-risk thresholds')],fontsize=8)
    fig.suptitle('192 REUSED traces: offline prefix accounting, not new closed-loop runs or calibrated stopping',fontsize=10)
    for ext in ('png','svg'):fig.savefig(base/f'cost_report_tradeoff.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    files=['cost_report_tradeoff.png','cost_report_tradeoff.svg']
    m=dict(source_sha256=hashlib.sha256((base/'analysis.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in files},
        scope='All4tracefamilies/12rules each; fixed points connected for readability only, not interpolated policies; descriptive, no independent uncertainty bars')
    (base/'figure_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')


if __name__=='__main__':main()
