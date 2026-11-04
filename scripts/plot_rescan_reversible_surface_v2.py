"""Descriptive paired results; no independent-object confidence interval."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/rescan_reversible_surface_v2');data=json.loads((root/'scores.json').read_text())
    methods=['frozen','hard_forward','joint_unrooted','rooted_forward','rooted_joint']
    names=['Frozen','Hard forward','Ordinary joint','Rooted forward','Rooted joint']
    strict=[];equiv=[];cases=[]
    for m in methods:
        rr=[r for r in data['summaries'] if r['period']==2 and r['method']==m]
        strict.append(sum(r['strict_correct'] for r in rr));equiv.append(sum(r['equivalent_correct'] for r in rr))
        cases.append([next(r['strict_fraction'] for r in data['rows'] if r['scene']=='scene_a' and r['period']==2 and r['method']==m and r['truth_id']==uid) for uid in [9,24]])
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
        y=np.arange(5)
        for ax,values,title,color in zip(axs[:2],[strict,equiv],['A  Strict identity','B  Allowed equivalent identity'],['#0072B2','#009E73']):
            ax.barh(y,values,color=color,height=.65)
            ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,37);ax.set_xlabel('Assets with >=80% point coverage (of 37)');ax.set_title(title,loc='left')
            for i,v in enumerate(values):ax.text(v+.3,i,str(v),va='center')
        im=axs[2].imshow(np.array(cases)*100,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        axs[2].set_yticks(y,names);axs[2].set_xticks([0,1],['Asset 9','Asset 24']);axs[2].set_title('C  Preselected cases: strict coverage',loc='left')
        for i in range(5):
            for j in range(2):
                v=100*cases[i][j];axs[2].text(j,i,f'{v:.1f}%',ha='center',va='center',color='white' if v>55 else 'black')
        fig.suptitle('Rescan capture 2 | Two familiar development spaces | Same native initialization',fontsize=12)
        fig.supxlabel('Descriptive paired outcomes; correlated assets. No displacement truth or independent-test claim.',fontsize=10)
        fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf')
        plt.close(fig)
    (root/'figure_data.json').write_text(json.dumps({'methods':methods,'strict':strict,'equivalent':equiv,'case_strict_fractions':cases,'denominator':37,
        'caption':'Five same-input prototype arms on capture2 of two familiar spaces; source-side updates differ. Counts are descriptive, not independent trials. Case selections were fixed before this run.'},indent=2))
    print('FIGURE WRITTEN')


if __name__=='__main__':main()
