from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_rigid_identity_bounds_v1 import read,save


def main():
    root=Path('results/visible_sampling_centroid_v1');out=root/'figures';out.mkdir(exist_ok=True);s=read(root/'scores.json')
    methods=['pair_only','ordinary_shared_nn','ordinary_rendered_pixels','fixed_centroid','noise_inflated_centroid','visible_centroid','visible_centroid_plus_scatter']
    labels=['Pair score','Shared surface NN','Rendered pixel score','Fixed cell centroid','Inflated centroid noise','Visible centroid','Visible centroid + scatter']
    e=np.array([[r['anchor_error_m']*1000 for r in s['rows'] if r['method']==m] for m in methods]);shift=np.array([r['diagnosis']['moments']['mean_shift_mm'] for r in s['rows'] if r['method']=='visible_centroid'])
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(2,1,figsize=(12.8,6.6),layout='constrained',gridspec_kw={'height_ratios':[1.7,1]})
        im=ax[0].imshow(e,cmap='viridis',vmin=0,vmax=30,aspect='auto')
        for i in range(len(methods)):
            for j in range(12):ax[0].text(j,i,f'{e[i,j]:.1f}',ha='center',va='center',color='black' if e[i,j]>18 else 'white',fontsize=9)
        ax[0].set_yticks(range(len(methods)),labels);ax[0].set_xticks(range(12),[f'{j:02d}' for j in range(12)]);ax[0].set_title('(a) Material-anchor error; same 3,965 hypotheses, target excluded from surface')
        fig.colorbar(im,ax=ax[0],label='Anchor error (mm)')
        ax[1].bar(range(12),shift,color='#0072B2');ax[1].set_xticks(range(12),[f'{j:02d}' for j in range(12)]);ax[1].set_ylabel('Predicted mean shift (mm)');ax[1].set_xlabel('Pair (one previously seen video; correlated development observations)');ax[1].set_ylim(0,max(shift)*1.25)
        for j in range(12):ax[1].text(j,shift[j]+.15,f'{shift[j]:.2f}',ha='center',fontsize=9)
        ax[1].set_title('(b) Visible vs full-cell center at selected pose; predicted bias, not measured motion')
        for a in ax:
            for v in [2.5,5.5,8.5]:a.axvline(v,color='grey',ls=':',lw=.8)
        fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    save(out/'data.json',dict(methods=methods,error_mm=e.tolist(),predicted_mean_shift_mm=shift.tolist(),scope='All12pairs, no exclusions/no inferential CIs. Shift is an internal model quantity.'))


if __name__=='__main__':main()
