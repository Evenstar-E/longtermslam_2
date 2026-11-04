from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_rigid_identity_bounds_v1 import read,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/joint_source_regret_v1');args=ap.parse_args()
    out=Path(args.output);dest=out/'figures';dest.mkdir(exist_ok=True);s=read(out/'scores.json');p=read(out/'episode_02/predictions.json')
    methods=['initial_pair_choices','joint_nn','joint_ray','worst_source','minimax_source_regret'];labels=['Independent initialization','Joint geometric mean','Joint ray mean','Worst source deletion','Minimax source regret']
    e=np.array([[r['anchor_error_m']*1000 for r in s['rows'] if r['method']==m] for m in methods]);deletions=np.array([p['rows'][p['choices'][m]]['loss']['deletion_losses'] for m in methods]);regrets=deletions-np.array(p['robust']['deletion_best_losses'])
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,1,figsize=(12.8,7),layout='constrained',gridspec_kw={'height_ratios':[1.1,1]})
        vmax=max(30,float(e.max()));im=axes[0].imshow(e,cmap='viridis',vmin=0,vmax=vmax,aspect='auto')
        for i in range(5):
            for j in range(12):axes[0].text(j,i,f'{e[i,j]:.1f}',ha='center',va='center',color='black' if e[i,j]>.6*vmax else 'white',fontsize=9)
        axes[0].set_yticks(range(5),labels);axes[0].set_xticks(range(12),[f'{j:02d}' for j in range(12)]);axes[0].set_title('(a) Fixed historical material-anchor error; all12pairs, same joint-state pool');fig.colorbar(im,ax=axes[0],label='Anchor error (mm)')
        for x in [2.5,5.5,8.5]:axes[0].axvline(x,color='grey',ls=':',lw=.8)
        colors=['#555555','#0072B2','#009E73','#E69F00','#CC79A7'];markers=['o','s','^','D','x']
        for i,(label,c,m) in enumerate(zip(labels,colors,markers)):axes[1].plot(range(4),regrets[i],label=label,color=c,marker=m)
        axes[1].set_xticks(range(4),['Drop history','Drop view1','Drop view2','Drop view3']);axes[1].set_ylabel('Regret in ray surrogate');axes[1].set_title('(b) Hard episode02: deletion-scenario regret is NOT physical displacement error');axes[1].legend(fontsize=8,ncol=2);axes[1].set_ylim(bottom=0)
        fig.savefig(dest/'comparison.png',dpi=180);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    save(dest/'data.json',dict(methods=methods,error_mm=e.tolist(),hard_episode_deletion_losses=deletions.tolist(),hard_episode_regrets=regrets.tolist(),scope='Same old single-video4episodes, correlated12pairs; no inferential CIs/no independent replication claim.'))


if __name__=='__main__':main()
