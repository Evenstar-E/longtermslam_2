import csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from comparative_evidence.estimation_v1 import reference_map,world_components
from inspection_decisions.projective_visibility_v1 import decode
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    out=Path('results/se3_visibility_v1');a=read(out/'analysis.json');audit=read(out/'audit.json');budget=read('results/se3_visibility_budget_audit_v1/analysis.json')
    mainrows=[r for r in a['rows'] if r['method']=='se3_rescore'];retained_counts=[]
    for sid in range(30):
        b=read(out/f'{sid:04d}_backend.json');retained_counts.append(len({str(r['matches']) for r in b['registration']['retained']}))
    save(out/'mechanism_audit.json',dict(status='descriptive no model changes',initial_assignment_counts=dict(Counter(r['initial_assignment_hypotheses'] for r in mainrows)),
        refined_assignment_counts=dict(Counter(r['refined_assignment_hypotheses'] for r in mainrows)),retained_actual_assignment_counts=dict(Counter(retained_counts)),
        retained_pose_counts=dict(Counter(r['retained_poses'] for r in mainrows)),late_selected_counts=dict(Counter(r['late_selected'] for r in mainrows)),
        conclusion='Main30conditions do NOT retain competing actual identities; pose-only alternatives. Partial-FOV fixture separate. Budget-matched ordinary prior removes candidate gain.'))
    with (out/'scene_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['case','family','condition','method','qualified','fp','badmotion','identity_not_ok','unknown','loss','rotation_error_deg','translation_error_m'])
        for r in a['rows']:w.writerow([r['scene'],r['family'],r['condition'],r['method'],r['metric']['qualified_tp'],r['metric']['fp'],r['metric']['bad_motion_reports'],r['metric']['identity_wrong_references'],r['unknown'],r['metric']['task_loss'],r['rotation_error_deg'],r['translation_error_m']])
    names=['se3_late_visibility','se3_rescore','ordinary_prior_score','ordinary_pixel_score','se3_frozen_identity','se3_cloud_only','translation_projective']
    labels=['Late visibility','Ordinary early pose','Ordinary same-budget prior*','Ordinary same-budget pixels*','Frozen identities','Cloud only','Translation only']
    colors=['#0072B2','#E69F00','#009E73','#56B4E9','#CC79A7','#777777','#D55E00']
    values={m:(budget['summaries']['all'][m] if m.startswith('ordinary_') else a['summaries']['all'][m]) for m in names}
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,key,title in zip(axs.flat[:3],['qualified_tp','fp','task_loss'],['Qualified changed instances (out of 30)','False change reports','Task loss (lower is better)']):
            vals=[values[m][key] for m in names];ax.barh(np.arange(7),vals,color=colors);ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlim(0,max(vals)*1.16)
            for i,v in enumerate(vals):ax.text(v+max(vals)*.012,i,f'{v:g}',va='center')
        ax=axs[1,1];x=np.arange(6);ax.axhline(0,color='#888888',lw=1)
        ax.plot(x,audit['primary_contrasts']['se3_rescore']['differences'],'s',color='#E69F00',label='vs early ordinary pose')
        ax.plot(x,budget['contrasts']['ordinary_prior_score']['differences'],'o',color='#009E73',label='vs same-budget prior*')
        ax.set(xlabel='Repeated world (one history view, SE3)',ylabel='Candidate minus control loss',xticks=x);ax.legend(fontsize=8)
        ax.set_title('Single-world gain disappears\nwith matched compute',fontsize=11)
        fig.suptitle('SE3 scope pilot: 12 worlds / 30 dependent conditions\n*POSTHOC budget controls, not confirmation; main identity hypotheses all identical',fontsize=13)
        fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    fixture=Path('results/partial_fov_identity_fixture_v1');raw=read(fixture/'0000_observed.json')['raw'];reg=read(fixture/'0000_registration.json');ev=read(fixture/'analysis.json')['rows'][0]
    h=decode(raw['history'][0]);q=decode(raw['current'][0]);refs=reference_map([h]);cs=world_components(q)
    with plt.rc_context({'font.size':9}):
        fig=plt.figure(figsize=(12,7),layout='constrained');gs=fig.add_gridspec(2,6)
        ax=fig.add_subplot(gs[0,:3]);ax.imshow(h['rgb'],interpolation='nearest');ax.set_title('Acquired history: wider view, 8 visible references');ax.set(xlabel='Image x (px)',ylabel='Image y (px)')
        ax=fig.add_subplot(gs[0,3:]);ax.imshow(q['rgb'],interpolation='nearest');ax.set_title('Acquired current: closer view, 4 visible components');ax.set(xlabel='Image x (px)',ylabel='Image y (px)')
        X=np.array([r['centre'] for r in refs]);Y=np.array([c['centre'] for c in cs])
        for k,branch in enumerate(reg['retained']):
            ax=fig.add_subplot(gs[1,2*k:2*k+2]);G=np.array(branch['G']);P=(Y-G[:3,3])@G[:3,:3]
            ax.scatter(X[:,0],X[:,1],marker='s',facecolors='none',edgecolors='#777777',s=80,label='historical references')
            ax.scatter(P[:,0],P[:,1],color='#0072B2',marker='x',s=45,label='current under hypothesis')
            for j,x in enumerate(X):ax.text(x[0]+.05,x[1]+.06,str(j),fontsize=8)
            ax.set(xlim=(-3.2,3.2),ylim=(-1.4,1.4),xlabel='Archive x (m)',ylabel='Archive y (m)');ax.set_aspect('equal')
            ax.set_title(f'Hypothesis {k+1}: score {branch["score"]:.4f}\nEVALUATOR ONLY: {ev["hypotheses"][k]["correct"]}/4 physical IDs correct',fontsize=9)
        fig.suptitle('Separate input-readiness fixture: three ACTUAL identity mappings from RGB-D\nNot task performance, not a new algorithm result; correct mapping ranks third in this example',fontsize=12)
        fig.savefig(out/'partial_fov_identity.png',dpi=180);fig.savefig(out/'partial_fov_identity.svg');plt.close(fig)
    print('Wrote main comparisons, 150-row CSV, explicit identity mechanism audit and actual partial-FOV fixture figure.')


if __name__=='__main__':main()
