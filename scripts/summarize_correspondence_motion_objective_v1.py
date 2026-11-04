"""Post-evaluation descriptive diagnostics and honest pilot figure; no fitting."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    out=Path('results/correspondence_motion_objective_v1');a=read(out/'analysis.json');audit=read(out/'audit.json')
    methods=a['config']['methods'];diagnosis=[]
    for row in a['rows']:
        sid=row['scene'];method=row['method'];data=read(out/f'{sid:04d}_observed.json')['evidence']
        r=read(out/f'{sid:04d}_{method}.json');truth=read(out/'scenes'/f'{sid:04d}'/'evaluation/truth.json')
        for rec in row['metric']['ownership_records']:
            j=rec['ref_id'];owner=rec['owner']
            if not (rec['label']==1 and owner>=0 and not truth['changed'][owner]):continue
            observations=[]
            for f,(frame,pairs) in enumerate(zip(data['frames'],r['best']['matches'])):
                for i,k in pairs:
                    if k==j:observations.append(dict(frame=f,shift_m=(np.array(frame['D'][i][j])-np.array(r['best']['translation'])).tolist(),
                        surface_residual_m=frame['S'][i][j],pixels=len(frame['pixels'][i])))
            gap=float(np.linalg.norm(np.array(observations[0]['shift_m'])-observations[1]['shift_m'])) if len(observations)==2 else None
            diagnosis.append(dict(scene=sid,method=method,reference=j,evaluation_owner=owner,identity_ok=rec['identity_ok'],
                fitted_delta_m=r['best']['delta'][j],observations=observations,cross_view_shift_gap_m=gap))
    save(out/'posthoc_diagnosis.json',dict(status='descriptive after evaluation; no algorithm changes',false_alarm_records=diagnosis,
        candidate_false_alarm_records=sum(r['method']=='shared_motion' for r in diagnosis),
        limitation='partial visibility is a plausible mechanism, not identified by a visibility intervention; scene13 threshold sensitivity is real'))
    with (out/'scene_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scene','regime','method','qualified_tp','fp','bad_motion','identity_not_ok','unknown','task_loss','pose_error_m'])
        for r in a['rows']:w.writerow([r['scene'],r['regime'],r['method'],r['metric']['qualified_tp'],r['metric']['fp'],r['metric']['bad_motion_reports'],r['metric']['identity_wrong_references'],r['unknown'],r['metric']['task_loss'],r['pose_error_m']])
    labels=['Shared motion','Independent profile','No amplitude cost','Cauchy multistart','Old capped joint']
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#777777']
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,7.7),layout='constrained')
        for ax,key,title in zip(axs.flat[:3],['qualified_tp','fp','task_loss'],['Qualified changes (out of 16)','False change reports (lower is better)','Total task loss (lower is better)']):
            values=[a['summaries']['all'][m][key] for m in methods];y=np.arange(5)
            ax.barh(y,values,color=colors,height=.65);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,fontsize=11)
            ax.set_xlim(0,max(values)*1.18+1e-3)
            for i,v in enumerate(values):ax.text(v+max(values)*.015,i,f'{v:g}',va='center')
        ax=axs[1,1];d=audit['contrasts']['independent_profile']['scene_loss_differences']
        ax.axhline(0,color='#777777',lw=1);ax.plot(range(16),d,'D',color='#0072B2',markersize=5)
        ax.set(xlabel='Scene family',ylabel='Candidate minus control task loss',xticks=range(0,16,2))
        ax.set_title('Paired loss vs independent profile\n1 improved / 1 worse / 14 tied',fontsize=11)
        fig.suptitle('Frozen motion-objective pilot: 16 fresh seed families\nShared evidence; exploratory, not independent layout confirmation',fontsize=13)
        fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg')
        plt.close(fig)
    print('Wrote figure, 80-row CSV and posthoc false-alarm diagnostics; no model changes.')


if __name__=='__main__':main()
