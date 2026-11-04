"""Descriptive same-input pool and dependency comparisons, no independence claim."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/provenance_pose_hypotheses_v1');methods=['selected_forward','pool_forward','pool_detached','pool_coupled']
    names=['Selected pose','Full pool, fixed source','Full pool, detached','Full pool, coupled']
    data={};payload={}
    for prefix in ['', 'nn_']:
        s=json.loads((root/(prefix+'scores.json')).read_text());rows=[]
        for m in methods:
            rr=[r for r in s['summaries'] if r['period']==2 and r['method']==m]
            rows.append([sum(x['strict_correct'] for x in rr),sum(x['equivalent_correct'] for x in rr)])
        data[prefix]=np.array(rows);payload['state' if not prefix else 'nn_readout']=rows
    sources=json.loads((root/'provenance_diagnosis.json').read_text())['source_corrections']
    source=np.array([[sum(r[k] for r in sources if r['method']==m) for k in ['equivalent_fixed_voxels','equivalent_damaged_voxels']] for m in methods[2:]])
    payload['source_fixed_damaged']=source.tolist()
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(1,3,figsize=(14,5.7),layout='constrained')
        for ax,prefix,title in zip(axs[:2],['','nn_'],['A  Primary: optimized ownership','B  Secondary: pointwise NN readout']):
            y=np.arange(4)
            ax.barh(y-.17,data[prefix][:,0],height=.32,label='Strict',color='#0072B2')
            ax.barh(y+.17,data[prefix][:,1],height=.32,label='Allowed equivalent',color='#E69F00',hatch='//')
            ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,37);ax.set_xlabel('Assets with >=80% original-point coverage / 37')
            ax.set_title(title,loc='left');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=8)
            for j in range(4):
                for k in range(2):ax.text(data[prefix][j,k]+.3,j+(-.17 if k==0 else .17),str(data[prefix][j,k]),va='center',fontsize=9)
        ax=axs[2];x=np.arange(2)
        ax.bar(x-.18,source[:,0],width=.34,label='Corrected',color='#009E73')
        ax.bar(x+.18,source[:,1],width=.34,label='Damaged',color='#CC79A7',hatch='xx')
        ax.set_xticks(x,['Detached','Coupled']);ax.set_ylabel('Source voxels (allowed-equivalence evaluation)');ax.set_title('C  Actual source-memory revisions',loc='left')
        ax.legend(fontsize=8)
        for j in range(2):
            for k in range(2):ax.text(j+(-.18 if k==0 else .18),source[j,k],str(source[j,k]),ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(1,source.max())*1.18)
        fig.suptitle('Two familiar Rescan development spaces | Shared finite pose pools | No displacement truth',fontsize=12)
        fig.supxlabel('Descriptive outcomes. Correlated assets/voxels; no independent-test or significance claim.',fontsize=10)
        fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
    (root/'figure_data.json').write_text(json.dumps({'methods':methods,**payload,'caption':'Primary coupled-state readout and secondary independent NN readout are prespecified, not selected by outcome.'},indent=2))
    print('FIGURE WRITTEN')


if __name__=='__main__':main()
