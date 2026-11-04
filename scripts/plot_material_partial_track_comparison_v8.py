"""Provisional research figure, all methods/conditions retained, one TRAIN scene."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    dest=Path('results/material_partial_track_fixed_measure_v8');roots=[Path('results/material_partial_track_transport_v7'),dest];methods=['ordinary_union','visibility_pair','history_union','shared_transport'];variants=['flat','rich','repeated','changed'];labels=['Ordinary union','Independent visibility','Append-only history','Revisable shared material'];colors=['#555555','#0072B2','#D55E00','#009E73'];hatches=['','//','..','xx'];source=[]
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharey=True);fig.subplots_adjust(left=.06,right=.99,bottom=.19,top=.76,wspace=.09)
        for stage,(root,ax) in enumerate(zip(roots,axes)):
            data=json.loads((root/'scores.json').read_text(encoding='utf-8'))['summary']
            for c,m in enumerate(methods):
                vals=[next(r['qualified'] for r in data if r['variant']==v and r['method']==m) for v in variants];bars=ax.bar(np.arange(4)+(c-1.5)*.19,vals,.18,label=labels[c],color=colors[c],hatch=hatches[c],edgecolor='black',linewidth=.5)
                for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.35,str(val),ha='center',va='bottom',fontsize=8)
                source.extend(dict(stage=stage,method=m,variant=v,qualified=value,total=24) for v,value in zip(variants,vals))
            ax.set(xticks=np.arange(4),xticklabels=['Flat','Rich','Repeated','Repainted'],ylim=(0,24),yticks=np.arange(0,25,4),title=['V7: pose-dependent denominator','V8: fixed observation measure'][stage]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylabel('Qualified tasks / 24');handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.98),ncol=2,frameon=False);fig.text(.5,.07,'One TRAIN scene; correlated objects and appearances. No confidence intervals or generalization claim.',ha='center',fontsize=10);fig.text(.5,.025,'Same inputs / final candidates within each stage; candidate endpoints and identity states may differ across stages.',ha='center',fontsize=9)
        fig.savefig(dest/'task_comparison.png',dpi=180,facecolor='white');fig.savefig(dest/'task_comparison.pdf',facecolor='white');plt.close(fig)
    with (dest/'figure_source.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    save(dest/'figure_provenance.json',dict(source_hashes={str(r/'scores.json'):sha(r/'scores.json') for r in roots},estimator='Strict qualified count: both IDs correct and all three temporal fixed-material P95 errors <= 0.05 m.',replication='One TRAIN scene, 24 correlated objects, 4 correlated appearance variants',transformations='Select final four methods from each complete score file; no exclusions; zero bars labeled.',uncertainty='None; tasks are not independent scenes',target='Internal research report; journal requirements unspecified',dimensions_inches=[12,5.2],dpi=180,alt='Two panels compare four methods in four appearances before and after fixed-denominator repair. Exact counts are printed on every bar, including zeros.'))

if __name__=='__main__':main()
