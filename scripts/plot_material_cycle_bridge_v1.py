from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_rigid_identity_bounds_v1 import read,save


def main():
    out=Path('results/material_cycle_bridge_v1');dest=out/'figures';dest.mkdir(exist_ok=True)
    s=read(out/'scores.json');d=read(out/'material_witness_diagnosis.json')
    methods=['old_pool_pair_only','pair_only','multiview_nn','pose_cycle','material_cycle','visible_material_cycle']
    labels=['Old pool: pair score','Shared pool: pair score','Multi-view NN','Pose cycle','Material cycle','Visible material cycle']
    values=np.array([[r['anchor_error_m']*1000 for r in s['rows'] if r['method']==m] for m in methods])
    count=np.array([sum(b['estimated_composed_matches'] for b in r['bridges']) for r in d['rows']]);valid=np.array([sum(b['reference_consistent_15mm'] for b in r['bridges']) for r in d['rows']])
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,1,figsize=(12.8,6.5),layout='constrained',gridspec_kw={'height_ratios':[1.4,1]})
        im=axes[0].imshow(values,cmap='viridis',vmin=0,vmax=40,aspect='auto')
        for i in range(6):
            for j in range(12):axes[0].text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='black' if values[i,j]>23 else 'white',fontsize=9)
        axes[0].set_yticks(range(6),labels);axes[0].set_xticks(range(12),[f'{j:02d}' for j in range(12)])
        axes[0].set_title('(a) Fixed historical material-anchor error; every pair retained')
        fig.colorbar(im,ax=axes[0],label='Anchor error (mm)')
        axes[1].bar(range(12),valid/count,color='#0072B2',label='Endpoint within 15 mm of reference-transformed position')
        axes[1].set_ylim(0,1.12);axes[1].set_yticks([0,.25,.5,.75,1]);axes[1].set_ylabel('Reference-consistent fraction');axes[1].set_xlabel('Pair (four episodes of one video; not independent trials)')
        axes[1].set_xticks(range(12),[f'{j:02d}' for j in range(12)])
        for j in range(12):axes[1].text(j,valid[j]/count[j]+.025,f'{valid[j]}/{count[j]}',ha='center',fontsize=9)
        axes[1].set_title('(b) Bridge material witnesses; counts are correlated, threshold is diagnostic')
        for ax in axes:
            for x in [2.5,5.5,8.5]:ax.axvline(x,color='grey',ls=':',lw=.8)
        fig.savefig(dest/'comparison.png',dpi=180);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    save(dest/'figure_data.json',dict(methods=methods,error_mm=values.tolist(),witness_count=count.tolist(),reference_consistent_count=valid.tolist(),scope='Development figure; no inferential confidence intervals, no independent sample-size claim. All12pairs, no exclusion. Endpoint consistency is not annotated material identity.'))


if __name__=='__main__':main()
