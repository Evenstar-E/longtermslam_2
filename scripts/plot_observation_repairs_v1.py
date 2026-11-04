"""Descriptive development figures; source data and correlation limits explicit."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    out=Path('results/ray_material_full_resolution_v1');r=read(out/'scores.json');root=Path('data/ycbineoat_motion_prepared_v2');dest=out/'figures';dest.mkdir(exist_ok=False)
    methods=['old_nn','ordinary_censored_depth','ordinary_censored_rgb','ordinary_overlap_rgb','beta_geometry','beta_material','ordinary_exact_beta_equivalent']
    labels=['Old NN + motion','Naive censored depth','Naive censored RGB-D','Ordinary overlap + RGB','Beta occupancy','Beta + coarse material','Exact ordinary equivalent']
    rows=[v for v in r['rows'] if v['scope']=='shared_pool'];pairs=sorted({v['pair'] for v in rows});data=np.array([[next(v['anchor_error_m']*1000 for v in rows if v['method']==m and v['pair']==p) for p in pairs] for m in methods])
    with (dest/'anchor_errors_mm.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['method']+pairs);w.writerows([[m]+list(d) for m,d in zip(methods,data)])
    with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(12,4.9),layout='constrained');im=ax.imshow(data,cmap='viridis',norm=LogNorm(vmin=.05,vmax=1000),aspect='auto')
        ax.set_yticks(range(len(methods)),labels);ax.set_xticks(range(len(pairs)),[p[-2:] for p in pairs]);ax.set_xlabel('Temporal pair index (same video, repeated historical frames)')
        for i in range(len(methods)):
            for j in range(len(pairs)):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=8,color='black' if data[i,j]>100 else 'white')
        fig.colorbar(im,ax=ax,label='Material-anchor error (mm), log color scale');ax.set_title('Identical 3,941-pose pool, all observed points; development comparison')
        for ext in ['png','pdf']:fig.savefig(dest/('all_pair_errors.'+ext),dpi=180)
        plt.close(fig)
        p=read(out/'pair_08/predictions.json');diag=next(v for v in r['diagnosis'] if v['pair']=='pair_08');select=['old_nn','ordinary_overlap_rgb','beta_material'];names=['Annotation reference','Old NN choice','Ordinary overlap choice','Beta/material choice'];states=[diag['GT_ray_diagnosis']]+[p['pool'][p['choices'][m]]['diagnosis'] for m in select]
        errors=[0]+[next(v['anchor_error_m']*1000 for v in rows if v['pair']=='pair_08' and v['method']==m) for m in select]
        fig=plt.figure(figsize=(12,6.8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.2,1]);ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);ax=fig.add_subplot(gs[1,:])
        for a,stem,title in [(ax1,p['pair']['history_stem'],'Historical RGB'),(ax2,p['pair']['current_stem'],'Current RGB')]:
            a.imshow(Image.open(root/'public/rgb'/(stem+'.png')));a.set_title(title+' (unaltered)');a.axis('off')
        left=np.zeros(4)
        for key,label,color,hatch in [('surface','Observed target depth','#0072B2',''),('free','Known target absence','#D55E00','//'),('unknown','Unseen / occluded','#B8B8B8','..')]:
            values=np.array([d[key]*100 for d in states]);ax.barh(range(4),values,left=left,color=color,label=label,hatch=hatch,edgecolor='white');left+=values
        ax.set_yticks(range(4),[f'{n}; anchor error {e:.1f} mm' for n,e in zip(names,errors)]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Bidirectional query fractions (%); same full observed point sets');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3)
        fig.suptitle('Pair 08: more apparent overlap can favor a physically wrong pose\nAnnotation pose is diagnostic only; never a proposal or initialization')
        for ext in ['png','pdf']:fig.savefig(dest/('pair08_surface_alias.'+ext),dpi=180)
        plt.close(fig)
    save(dest/'manifest.json',dict(source_sha256=sha(out/'scores.json'),script_sha256=sha(__file__),units='mm and fractions',scope='12 correlated temporal pairs of one video, no independent error bars. No pose exclusions. Log color scale explicitly labeled. Reference pose used only in post-seal diagnosis. Raw RGB displayed without image adjustment. General research figure, no publisher specification claimed.'))

if __name__=='__main__':main()
