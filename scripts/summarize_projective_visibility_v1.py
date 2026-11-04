"""Frozen-result figures and post-evaluation failure attribution; no fitting."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from comparative_evidence.estimation_v1 import reference_map,world_components
from inspection_decisions.projective_visibility_v1 import decode,component_mask,triangles_from_mask,rasterize
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    out=Path('results/projective_visibility_v1');a=read(out/'analysis.json');audit=read(out/'audit.json');cfg=a['config'];missing=[];repairs=[]
    for r in a['rows']:
        if r['method']=='joint_censored' and r['unknown']:
            sid=r['scene'];root=out/'scenes'/f'{sid:04d}';b=read(out/f'{sid:04d}_backend.json')
            for asset,rec in zip(read(out/f'{sid:04d}_joint_censored.json')['prediction']['assets'],r['metric']['ownership_records']):
                if asset['label']>=0:continue
                observed=[]
                for f,frame in enumerate(b['legacy']['frames']):
                    with np.load(root/'evaluation'/f"{frame['token']}_owner.npz") as z:owners=z['owner'].ravel()
                    for i,pixels in enumerate(frame['pixels']):
                        purity=float(np.mean(owners[pixels]==rec['owner']))
                        if purity>=.8:observed.append(dict(frame=f,component=i,purity=purity,
                            initially_selected=(i,asset['ref_id']) in list(map(tuple,b['initial']['best']['matches'][f])),
                            observed_pair_D=b['legacy']['frames'][f]['D'][i][asset['ref_id']]))
                missing.append(dict(scene=sid,reference=asset['ref_id'],evaluation_owner=rec['owner'],current_observations=observed,
                    explanation='Observed target was rejected in common initial association, so no projective fit was attempted. No truth fed to actor.'))
        if r['method']=='legacy_shared' and r['metric']['fp']:
            sid=r['scene'];truth=read(out/'scenes'/f'{sid:04d}'/'evaluation/truth.json')
            for rec in r['metric']['ownership_records']:
                if rec['label']!=1 or rec['owner']<0 or truth['changed'][rec['owner']]:continue
                j=rec['ref_id'];preds={m:read(out/f'{sid:04d}_{m}.json')['prediction']['assets'][j] for m in ['legacy_shared','independent_censored','joint_censored']}
                repairs.append(dict(scene=sid,reference=j,evaluation_owner=rec['owner'],predictions=preds))
    same=all(all(x[k]==y[k] for k in ['qualified_tp','fp','bad_motion_reports','task_loss','identity_wrong_references']) and
             next(r['unknown'] for r in a['rows'] if r['scene']==sid and r['method']=='joint_censored')==next(r['unknown'] for r in a['rows'] if r['scene']==sid and r['method']=='independent_censored')
             for sid in range(20) for x,y in [(next(r['metric'] for r in a['rows'] if r['scene']==sid and r['method']=='joint_censored'),next(r['metric'] for r in a['rows'] if r['scene']==sid and r['method']=='independent_censored'))])
    save(out/'posthoc_diagnosis.json',dict(status='Frozen predictions only; no changes to model',same_task_endpoints_all20=bool(same),
        repaired_false_alarm_instances=repairs,missing_instances=missing,interpretation='Ordinary projective backend explains gain; joint increment not established.'))
    with (out/'scene_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scene','family','condition','method','qualified_tp','fp','bad_motion','identity_not_ok','unknown','task_loss'])
        for r in a['rows']:w.writerow([r['scene'],r['family'],r['condition'],r['method'],r['metric']['qualified_tp'],r['metric']['fp'],r['metric']['bad_motion_reports'],r['metric']['identity_wrong_references'],r['unknown'],r['metric']['task_loss']])
    methods=cfg['methods'];labels=['Joint censored','Per-frame censored','Joint no censor','Joint depth only','Legacy shared','Cauchy multistart'];colors=['#0072B2','#E69F00','#009E73','#CC79A7','#777777','#56B4E9']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,key,title in zip(axs.flat[:3],['qualified_tp','fp','task_loss'],['Qualified changed instances (out of 20)','False change reports (all scenes)','Task loss (all scenes, lower is better)']):
            vals=[a['summaries']['all'][m][key] for m in methods];y=np.arange(6)
            ax.barh(y,vals,color=colors);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlim(0,max(vals)*1.17)
            for i,v in enumerate(vals):ax.text(v+max(vals)*.015,i,f'{v:g}',va='center')
        ax=axs[1,1];x=np.arange(8);d=audit['contrasts']['legacy_shared']['occluded_differences'];p=audit['contrasts']['independent_censored']['occluded_differences']
        ax.axhline(0,color='#777777',lw=1);ax.plot(x,d,'s',color='#0072B2',label='vs legacy shared');ax.plot(x,p,'o',color='#E69F00',label='vs same-censor ordinary')
        ax.set(xlabel='Paired occluded world',ylabel='Candidate minus control task loss',xticks=x);ax.legend(fontsize=8)
        ax.set_title('No joint gain over ordinary projective fitting\nPrimary contrast uses 8 worlds',fontsize=11)
        fig.suptitle('Projective observation repair: 12 worlds / 20 rendered scenes / 120 outputs\nClear/occluded pairs are correlated; exploratory, not independent confirmation',fontsize=13)
        fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
        # First repaired legacy false-positive instance: scene1/reference5, declared posthoc.
        sid=repairs[0]['scene'];j=repairs[0]['reference'];raw=read(out/f'{sid:04d}_observed.json')['raw'];b=read(out/f'{sid:04d}_backend.json')
        hs=[decode(o) for o in raw['history']];qs=[decode(o) for o in raw['current']];refs=reference_map(hs);patches=[]
        for index,o in enumerate(hs):
            for i,c in enumerate(world_components(o)):
                chosen=i if index==0 else min((np.linalg.norm(c['centre']-r['centre']),k) for k,r in enumerate(refs) if c['group']==r['group'])[1]
                if chosen==j:patches.append(triangles_from_mask(o['depth'],o['nominal_T'],component_mask(o,c),raw['sensor'],cfg))
        triangles=np.concatenate(patches);diag=next(d for d in b['diagnostics'] if d['reference']==j)
        old=np.array(b['initial']['best']['translation'])+b['initial']['best']['delta'][j];new=np.array(diag['fits']['joint_censored'][0]['d'])
        fig,axs=plt.subplots(2,2,figsize=(8,7),layout='constrained')
        for f,i in diag['observations']:
            o=qs[f];c=world_components(o)[i];mask=component_mask(o,c);pold=np.isfinite(rasterize(triangles,o['nominal_T'],old,raw['sensor']));pnew=np.isfinite(rasterize(triangles,o['nominal_T'],new,raw['sensor']))
            yy,xx=np.where(mask|pold|pnew);loy=max(0,yy.min()-4);hiy=min(mask.shape[0],yy.max()+5);lox=max(0,xx.min()-4);hix=min(mask.shape[1],xx.max()+5)
            crop=(slice(loy,hiy),slice(lox,hix));extent=(lox,hix-1,hiy-1,loy)
            for col in [0,1]:axs[f,col].imshow(o['rgb'][crop],interpolation='nearest',extent=extent);axs[f,col].set(xlabel='Image x (px)',ylabel='Image y (px)')
            axs[f,0].set_title(f'Observed C{f}: untouched RGB crop',fontsize=10)
            axs[f,1].contour(np.arange(lox,hix),np.arange(loy,hiy),pold[crop],levels=[.5],colors=['#CC79A7'],linewidths=1.4)
            axs[f,1].contour(np.arange(lox,hix),np.arange(loy,hiy),pnew[crop],levels=[.5],colors=['#0072B2'],linewidths=1.4)
            axs[f,1].set_title('Known patch: magenta=old; blue=projective',fontsize=9)
        fig.suptitle(f'POSTHOC repaired static instance: scene {sid}, anonymous reference {j}\nC0 foreground hides most surface; C1 supplies complementary evidence',fontsize=11)
        fig.savefig(out/'repaired_case.png',dpi=180);fig.savefig(out/'repaired_case.svg');plt.close(fig)
    print('Wrote comparisons, raw-RGB projection overlay, 120-row CSV and posthoc attribution; no fitting.')


if __name__=='__main__':main()
