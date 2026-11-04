"""Raw development results; all objects/pairs retained, no independence assumption."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    out=Path('results/rigid_identity_cached_v1');scores=read(out/'scores.json');table=[]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        methods=['pair_one_to_one','pair_many_independent_pose','ordinary_common_rigid','ordinary_budget_filled','bounded_candidate','mass_cut_singleton'];names=['One-to-one pair fits','Independent many-to-one','Ordinary shared rigid pose','Ordinary, filled ICP budget','Bounded joint candidate','Singleton energy sets']
        rows=[next(r for r in scores['summary'] if r['scene']=='all' and r['method']==m) for m in methods];y=np.arange(len(rows));g=np.array([r['correct'] for r in rows]);w=np.array([r['wrong'] for r in rows]);u=34-g-w
        axes[0].barh(y,g,color='#0072B2',label='Correct');axes[0].barh(y,w,left=g,color='#D55E00',hatch='//',label='Wrong');axes[0].barh(y,u,left=g+w,color='#CCCCCC',hatch='..',label='Unresolved')
        for j,v in enumerate(g):axes[0].text(v+.15,j,str(v),va='center')
        axes[0].axvline(30,color='black',linestyle='--',label='Original Rescan: 30');axes[0].set(yticks=y,yticklabels=names,xlim=(0,35),xlabel='Persistent objects (34, two development spaces)',title='A  Actual predicted masks/classes');axes[0].invert_yaxis();axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=9)
        for scene in ['scene_a','scene_c']:
            for folder in sorted((out/scene).glob('class_*')):
                b=read(folder/'bounded.json');whole=read(Path('results/rigid_identity_cuts_v1_completed')/scene/(folder.name+'.json'));mass=read(Path('results/rigid_identity_mass_cuts_v1')/scene/(folder.name+'.json'))
                table.append({'label':scene[-1]+':'+folder.name.split('_')[1],'UB':b['best']['energy'],'base':b['last_lower_bound'],'whole':whole['cut_master']['lower_bound'],'mass':mass['cut_master']['lower_bound']})
        x=np.arange(len(table));axes[1].plot(x,[r['UB'] for r in table],'kx',label='Feasible upper bound');axes[1].plot(x,[r['base'] for r in table],'o',color='#0072B2',label='Base lower bound');axes[1].plot(x,[r['whole'] for r in table],'s',markerfacecolor='none',color='#D55E00',label='Whole-fragment cuts');axes[1].plot(x,[r['mass'] for r in table],'^',markerfacecolor='none',color='#009E73',label='Point-mass cuts')
        axes[1].set(xticks=x,xticklabels=[r['label'] for r in table],ylim=(0,.45),ylabel='Class-normalized surrogate energy',xlabel='Space : semantic class (compare within each class)',title='B  Same pose leaves, objective and upper bound');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=9)
        for ext in ['png','pdf']:fig.savefig(out/f'algorithm_comparison.{ext}',dpi=200)
        plt.close(fig)
        motion=Path('results/ycbineoat_motion_v1');ms=read(motion/'scores.json');diagnosis=read(motion/'energy_diagnosis.json');fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for method,color,marker in [('centroid_icp','#0072B2','o'),('fgr_gicp','#D55E00','s'),('more_pairwise','#009E73','^'),('bounded_candidate','#000000','x')]:
            rr=sorted([r for r in ms['rows'] if r['method']==method],key=lambda r:r['pair']);axes[0].plot(np.arange(12),[100*r['anchor_error_m'] for r in rr],marker=marker,color=color,label=method,linewidth=.8)
        axes[0].axhline(1,color='gray',linestyle='--');axes[0].set(yscale='log',xlabel='Frozen temporal pair (one sequence)',ylabel='Reference anchor error (cm, log scale)',title='C  Independent annotation reference');axes[0].legend(fontsize=8)
        e=ms['surrogate_energy_falsification'];axes[1].scatter([r['candidate_UB']+.01 for r in e],[r['GT_pose_energy'] for r in e],color='#0072B2');axes[1].plot([0,.25],[0,.25],'k--');axes[1].set(xlim=(0,.25),ylim=(0,.25),xlabel='Candidate UB + 0.01',ylabel='Reference pose energy',title='D  Energy region can exclude reference pose');axes[1].annotate('pair 08',xy=(e[8]['candidate_UB']+.01,e[8]['GT_pose_energy']),xytext=(.12,.225),arrowprops={'arrowstyle':'->'})
        for ext in ['png','pdf']:fig.savefig(motion/f'motion_control.{ext}',dpi=200)
        plt.close(fig)
        pred=read(motion/'pair_08/predictions.json');root=Path('data/ycbineoat_motion_prepared_v2');fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,key,title in [(axes[0],'history_stem','Historical RGB'),(axes[1],'current_stem','Current RGB')]:ax.imshow(Image.open(root/'public/rgb'/(pred['pair'][key]+'.png')));ax.set_title(title);ax.axis('off')
        rr=[r for r in diagnosis['rows'] if r['pair']=='pair_08' and r['method'] in ['GT_reference','candidate']];labels=['Reference pose','Wrong estimate'];bottom=np.zeros(2)
        for key,color,label in [('geometry_term','#0072B2','Observed-surface distance'),('appearance_term','#999999','Appearance'),('motion_regularizer','#D55E00','Motion regularizer')]:
            v=np.array([r[key] for r in rr]);axes[2].bar(labels,v,bottom=bottom,color=color,label=label);bottom+=v
        axes[2].set(ylim=(0,.25),ylabel='Same surrogate energy',title='Reference gets worse score');axes[2].legend(fontsize=8,loc='upper right')
        fig.savefig(motion/'energy_failure_case.png',dpi=200);fig.savefig(motion/'energy_failure_case.pdf');plt.close(fig)
    save(out/'figure_source_data.json',{'point_summary':rows,'class_bounds':table,'motion_summary':ms['summary'],'all_pair_energy':e,'pair08_components':rr})
    save(out/'figure_provenance.json',{'scope':'Provisional research review, no publisher-specific specification','samples':'34 persistent objects+6new in two already observed spaces;12 correlated temporal pairs from one manipulation sequence, not independent scenes','uncertainty':'Raw counts and pair values, no CI or significance claim','image_processing':'Unmodified original RGB displayed as panels; no retouching, contrast or pixel alteration','same_proposal_control':'10/12 within1cm with same median as candidate; candidate success not a new estimator gain','inputs_sha256':{str(p):sha(p) for p in [out/'scores.json',motion/'scores.json',motion/'energy_diagnosis.json']}})

if __name__=='__main__':main()
