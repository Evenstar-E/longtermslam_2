"""Frozen-result summary and explicitly posthoc pre-gate diagnostics."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspection_decisions.correspondence_joint_v1 import report
from comparative_evidence.evaluation_v3 import evaluate
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    out=Path('results/partial_surface_observation_v1');a=read(out/'analysis.json');diagnostics={};errors={};ratios={}
    for method in ['surface_directional','surface_isotropic']:
        rows=[];records=[];values=[]
        for sid in range(20):
            r=read(out/f'{sid:04d}_{method}.json');data=read(out/f'{sid:04d}_backends.json')['surface']
            # Reporting-gate removal is DIAGNOSTIC ONLY, not a new deployed method or tuned threshold.
            pred=report(data,r['best'],r['branches']);metric=evaluate(out/'scenes'/f'{sid:04d}',pred)
            source=next(x for x in a['rows'] if x['scene']==sid and x['method']==method)
            rows.append(dict(scene=sid,condition=source['condition'],metric=metric))
            truth=read(out/'scenes'/f'{sid:04d}'/'evaluation/truth.json')
            eigen=np.array(r['surface_information_eigenvalues']);values.extend((eigen[:,0]/np.maximum(eigen[:,-1],1e-12)).tolist())
            for rec in metric['ownership_records']:
                owner=rec['owner'];j=rec['ref_id']
                if owner<0 or not truth['changed'][owner]:continue
                records.append(dict(scene=sid,condition=source['condition'],reference=j,owner=owner,true_delta=truth['delta'][owner],
                    estimated_delta=r['best']['delta'][j],error_m=rec['motion_error_m']))
        summary={k:sum(r['metric'][k] for r in rows) for k in ['qualified_tp','fp','bad_motion_reports','task_loss']}
        diagnostics[method]=dict(status='POSTHOC_PRE_GATE_ONLY_NOT_ACTOR_RESULT',summary=summary,rows=rows,changed_errors=records,
            information_ratio_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist())
        errors[method]=records;ratios[method]=values
    save(out/'posthoc_diagnosis.json',dict(status='No model/parameter/actor change; pre-gate diagnostics only',diagnostics=diagnostics,
        limitation='Low local plane curvature does not establish unobservability of finite surface outlines or full RGB-D task.'))
    with (out/'scene_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scene','family','condition','method','qualified_tp','fp','bad_motion','identity_not_ok','unknown','task_loss'])
        for r in a['rows']:w.writerow([r['scene'],r['family'],r['condition'],r['method'],r['metric']['qualified_tp'],r['metric']['fp'],r['metric']['bad_motion_reports'],r['metric']['identity_wrong_references'],r['unknown'],r['metric']['task_loss']])
    labels=['Directional surface','Isotropic surface','Trimmed point ICP','Legacy shared motion','Cauchy multistart'];methods=a['config']['methods']
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#777777']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');y=np.arange(5)
        for ax,key,title in [(axs[0,0],'qualified_tp','Qualified changes: occluded worlds (out of 8)'),(axs[0,1],'task_loss','Task loss: occluded worlds (lower is better)')]:
            vals=[a['summaries']['occluded'][m][key] for m in methods]
            ax.barh(y,vals,color=colors);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,fontsize=11)
            ax.set_xlim(0,max(vals)*1.15)
            for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:g}',va='center')
        ax=axs[1,0];vals=[a['summaries']['all'][m]['unknown'] for m in methods]
        ax.barh(y,vals,color=colors);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,180)
        ax.set_title('UNKNOWN reports: all 160 asset instances',fontsize=11)
        for i,v in enumerate(vals):ax.text(v+3,i,str(v),va='center')
        ax=axs[1,1]
        for condition,color,marker in [('clear','#0072B2','o'),('occluded','#D55E00','s'),('normal','#009E73','^')]:
            x=[r['error_m'] for r in errors['surface_isotropic'] if r['condition']==condition]
            z=[r['error_m'] for r in errors['surface_directional'] if r['condition']==condition]
            ax.scatter(x,z,color=color,marker=marker,label=condition,s=32)
        mx=max(r['error_m'] for rs in errors.values() for r in rs)*1.1
        ax.plot([0,mx],[0,mx],color='#777777',lw=1);ax.axhline(.1,color='#aaaaaa',ls='--');ax.axvline(.1,color='#aaaaaa',ls='--')
        ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='Isotropic surface error (m)',ylabel='Directional surface error (m)')
        ax.set_title('PRE-GATE diagnostic only: changed displacement\n20 correlated instances, not 20 independent worlds',fontsize=10);ax.legend(fontsize=8)
        fig.suptitle('Partial-surface mechanism pilot FAILED: normal-only constraints discard useful geometry\n12 world families / 20 rendered scenes / 100 shared-evidence outputs',fontsize=13)
        fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    print('Wrote figure / CSV / posthoc pre-gate diagnostics. No model changes.')


if __name__=='__main__':main()
