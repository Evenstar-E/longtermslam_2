"""Actual graph aggregate diagram and descriptive labels; not performance plots."""
import collections,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from inspection_decisions.graph_dataset import read_graph,digest


def main():
    base=Path('results/evidence_graph_dataset_v1');a=json.loads((base/'analysis.json').read_text())
    source=base/'shard_0'/'inputs'/'states'/'0004_h01.json.gz';g=read_graph(source)
    nc=collections.Counter(n['kind'] for n in g['nodes']);ec=collections.Counter(e['kind'] for e in g['edges'])
    history=sum(r['epoch']=='history' for r in g['audit']['patch_sources']);current=sum(r['epoch']=='current' for r in g['audit']['patch_sources'])
    observed=sum(r['observed'] for r in g['audit']['nuisance_keys'])
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax=axes[0];positions=dict(asset=(0,2),history=(-1.3,.9),current=(1.3,.9),nuisance=(0,-.1),action=(0,-1.4))
    labels=dict(asset=f"Anonymous assets\nn={nc['asset']}",history=f'Observed history patches\nn={history}',
        current=f'Observed current patches\nn={current}',nuisance=f"Nuisance variables n={nc['nuisance']}\n{observed} observed / {nc['nuisance']-observed} prior",
        action=f"Available actions incl. STOP\nn={nc['action']}")
    for src,dst,color,curve in [('history','asset','#777777',.1),('current','asset','#777777',-.1),
        ('history','current','#009E73',.15),('current','history','#009E73',.15),
        ('history','nuisance','#E69F00',-.05),('current','nuisance','#E69F00',.05),
        ('action','nuisance','#E69F00',.05),('action','history','#0072B2',-.3),('action','current','#0072B2',.3)]:
        ax.add_patch(FancyArrowPatch(positions[src],positions[dst],connectionstyle=f'arc3,rad={curve}',
            arrowstyle='-|>',mutation_scale=13,color=color,lw=1.7,shrinkA=32,shrinkB=33))
    for key,(x,y) in positions.items():
        ax.text(x,y,labels[key],ha='center',va='center',fontsize=9,
            bbox=dict(boxstyle='round,pad=.55',facecolor='#F4F4F4',edgecolor='#333333'))
    ax.set_xlim(-2.5,2.5);ax.set_ylim(-2.05,2.5);ax.axis('off')
    ax.set_title('Actual scene4 after H0,H1,C0; aggregated nodes')
    ax.text(0,-1.95,'Selected relations shown; covariance / action-pair edges also exist',ha='center',fontsize=8)
    ax=axes[1];ax.axis('off')
    rows=[[k.replace('_',' '),str(v)] for k,v in sorted(ec.items())]
    table=ax.table(cellText=rows,colLabels=['All edge types','Count'],colWidths=[.8,.2],loc='center',cellLoc='left')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.9)
    ax.set_title(f"Exact counts: {len(g['nodes'])} nodes / {len(g['edges'])} edges")
    fig.suptitle('Graph expresses held geometry and approximate source correlations; no future utility or truth in input',fontsize=10)
    for ext in ('png','svg'):fig.savefig(base/f'graph_example.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    splits=list(a['splits']);x=np.arange(len(splits));fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,(key,label,color) in enumerate([('negative','Negative','#D55E00'),('zero','Zero','#999999'),('positive','Positive','#009E73')]):
        ax.bar(x+(j-1)*.24,[a['splits'][s][key] for s in splits],width=.24,color=color,label=label)
    ax.set_xticks(x,[s.replace('pipeline_','') for s in splits]);ax.set_ylim(bottom=0)
    ax.set_ylabel('Fixed action pairs (128 per split)');ax.legend(frameon=False)
    ax.set_title('Realized pair interaction labels;16 families/split, not algorithm or expected-value gains',fontsize=10)
    for ext in ('png','svg'):fig.savefig(base/f'pair_labels.{ext}',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    files=['graph_example.png','graph_example.svg','pair_labels.png','pair_labels.svg']
    (base/'figure_manifest.json').write_text(json.dumps(dict(source_graph=str(source),source_graph_sha256=digest(source),
        analysis_sha256=digest(base/'analysis.json'),script_sha256=digest(__file__),
        outputs={p:digest(base/p) for p in files},scope='Actual aggregate graph and allpairlabelsigncounts;noefficacyclaims/independentuncertaintybars'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
