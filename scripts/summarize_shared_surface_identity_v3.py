from pathlib import Path
from scripts.run_provenance_pose_hypotheses_v1 import read,save
import numpy as np


def main():
    roots=[Path('results/shared_surface_identity_likelihood_v3'),Path('results/shared_surface_identity_map_control_v3')];rows=[];sets={}
    diagnosis=read(roots[0]/'ownership_diagnosis.json')['rows']
    for root in roots:
        cfg=read(root/'config.json');runs=read(root/'runs.json')
        for method in cfg['methods']:
            row={'method':method}
            for prefix in ['','supported_','history_','nn_']:
                data=read(root/(prefix+'scores.json'));rr=[r for r in data['summaries'] if r['period']==2 and r['method']==method]
                row[prefix+'strict']=sum(r['strict_correct'] for r in rr);row[prefix+'equivalent']=sum(r['equivalent_correct'] for r in rr)
                if prefix=='':sets[method]=sorted(f"{r['scene']}:{r['truth_id']}" for r in data['rows'] if r['method']==method and r['period']==2 and r['strict_persistent'] and r['strict_correct'])
            d=[r for r in diagnosis if r['method']==method and r['period']==1]
            for key in ['expected_correct_identity_mass','expected_wrong_identity_mass','expected_outlier_mass','expected_newcomponent_correct_mass','expected_newcomponent_wrong_mass','high_confidence_mass','high_confidence_wrong_mass']:row['past_'+key]=sum(r[key] for r in d)
            row['elapsed_s']=sum(r['elapsed_s'] for r in runs if r['method']==method);row['kernel_evaluations']=sum(r['kernel_evaluations'] for r in runs if r['method']==method)
            rows.append(row)
    result={'rows':rows,'strict_sets':sets,'vb_vs_map_gain':sorted(set(sets['shared_vb'])-set(sets['shared_map'])),'vb_vs_map_loss':sorted(set(sets['shared_map'])-set(sets['shared_vb'])),'scope':'Descriptive two familiar spaces, same full finite pools, one pass, no displacement truth.'}
    save(roots[0]/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['frozen_shared','shared_mle','shared_map','shared_vb'];rr=[next(r for r in rows if r['method']==m) for m in order]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(1,3,figsize=(14,4.8),layout='constrained');x=np.arange(4)
        for j,(key,title) in enumerate([('strict','Identity success / 37'),('history_strict','Identity + history support / 37')]):
            ax[j].bar(x,[r[key] for r in rr],color=['#999999','#0072B2','#009E73','#E69F00']);ax[j].set_ylim(0,37);ax[j].set_title(title,loc='left')
            for k,r in enumerate(rr):ax[j].text(k,r[key]+.4,str(r[key]),ha='center')
        ax[2].bar(x,[r['past_expected_wrong_identity_mass'] for r in rr],color='#CC79A7',hatch='xx');ax[2].set_title('Past wrong-identity responsibility mass',loc='left')
        for a in ax:a.set_xticks(x,['Frozen','EM','MAP','VB'])
        fig.suptitle('Normalized shared-surface models | Same finite pose pools | Two DEV spaces')
        fig.supxlabel('Historical support is a fit diagnostic; responsibilities are not calibrated physical measurement uncertainty.')
        fig.savefig(roots[0]/'comparison.png',dpi=180);fig.savefig(roots[0]/'comparison.pdf');plt.close(fig)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
