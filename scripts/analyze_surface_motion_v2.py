"""Motion accuracy, selective risk, scene-cluster intervals and mainline audit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.surface_motion_v2 import stable_pose_information,fit_motion
from scripts.partial_surface_estimation import signed_box_distance
from scripts.joint_estimation import transform


def entries(row):
    correct=np.array(row['base_assignment'])==row['truth_assignment'];tc=np.array(row['truth_changed']);result=[]
    for i,f in row['fits'].items():
        i=int(i)
        if not (correct[i] and tc[i]):continue
        result.append(dict(scene=row['scene'],cell=row['cell'],error=float(np.linalg.norm(np.array(f['delta'])-row['truth_delta'][i])),near_bound=f['near_bound'],conditional_rank=f['conditional_rank'],marginal_rank=f['marginal_rank']))
    return result


def aggregate(rows):
    e=[e for r in rows for e in entries(r)];errors=np.array([x['error'] for x in e]);total=sum(sum(r['truth_changed']) for r in rows);fits=[f for r in rows for f in r['fits'].values()]
    result=dict(cases=len(rows),detected_true_changes=total,correct_positive_fits=len(e),median_vector_error_m=float(np.median(errors)),p90_vector_error_m=float(np.quantile(errors,.9)),maximum_vector_error_m=float(errors.max()),error_over_10cm=int((errors>.1).sum()),near_bound_true_positive=sum(x['near_bound'] for x in e),near_bound_all=sum(f['near_bound'] for f in fits),all_change_motion_success=float((errors<=.1).sum()/max(total,1)),median_fit_ms=float(np.median([r['seconds']*1000 for r in rows])))
    for mode in ('conditional','marginal'):
        accepted=[x for x in e if x[mode+'_rank']==3]
        result[mode+'_accepted_count']=len(accepted);result[mode+'_accepted_risk_over_10cm']=sum(x['error']>.1 for x in accepted)/max(len(accepted),1)
    return result


def main():
    out=Path('results/surface_motion_v2');cfg=json.loads((out/'frozen_config.json').read_text());manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()];checks=0
    for key in ('source_sha256','input_sha256'):
        for name,h in manifest[key].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h;checks+=1
    assert len(rows)==576 and len({(r['scene'],r['method']) for r in rows})==576;checks+=2
    for row in rows:
        base=manifest['base_predictions'][str(row['scene'])]
        assert row['base_assignment']==base['prediction_assignment'];assert row['base_changed']==base['prediction_changed'];checks+=2
        assert set(map(int,row['fits']))==set(np.flatnonzero(np.array(row['base_changed'])&row['base_known']));checks+=1
        for f in row['fits'].values():
            assert np.isfinite(f['delta']).all();assert f['marginal_rank']<=f['conditional_rank'];checks+=2
    # Reproduce the first fitting case in each split (all three methods).
    for split in ('reused','fresh'):
        sid=min(r['scene'] for r in rows if r['split']==split and r['fits']);base=manifest['base_predictions'][str(sid)]
        source=Path('results/partial_surface_v1/cases') if split=='reused' else out/'cases';obs=load_input(source/f'{sid:04d}_input.npz');T=np.array(base['T']);A=np.array(base['prediction_assignment']);bcfg=json.loads((out/'base_config.json').read_text())
        r=np.array([np.median(np.abs(signed_box_distance(transform(q['points'],T),obs['bounds'][j]))) for q,j in zip(obs['query'],A)]);H=stable_pose_information(obs,T,A,1/(1+(r/bcfg['group_scale_m'])**4))
        for method in cfg['methods']:
            old=next(r for r in rows if r['scene']==sid and r['method']==method);i=min(map(int,old['fits']));new=fit_motion(obs,T,A,i,cfg,method,H)
            np.testing.assert_allclose(new['delta'],old['fits'][str(i)]['delta'],atol=1e-9);checks+=1
    summary={split:{m:aggregate([r for r in rows if r['method']==m and r['split']==split]) for m in cfg['methods']} for split in ('reused','fresh')}
    # Fresh-only scene-cluster bootstrap; predefined 8 factorial cell sizes retained.
    fresh={m:[r for r in rows if r['method']==m and r['split']=='fresh'] for m in cfg['methods']};rng=np.random.default_rng(221023);intervals={}
    for control in ('one_sided','bidirectional'):
        bymethod={m:{r['scene']:entries(r) for r in fresh[m]} for m in (control,'visible_bidirectional')};diff=[]
        for _ in range(3000):
            ids=np.concatenate([rng.choice([r['scene'] for r in fresh[control] if r['cell']==c],size=8,replace=True) for c in range(8)])
            values={m:np.array([v['error'] for sid in ids for v in bymethod[m][int(sid)]]) for m in bymethod}
            diff.append([np.quantile(values['visible_bidirectional'],.9)-np.quantile(values[control],.9),(values['visible_bidirectional']>.1).mean()-(values[control]>.1).mean()])
        intervals[control]=dict(contrast='visible_bidirectional minus '+control,p90_difference_95_ci_m=np.quantile(np.array(diff)[:,0],[.025,.975]).tolist(),bad_rate_difference_95_ci=np.quantile(np.array(diff)[:,1],[.025,.975]).tolist())
    # Outcome-selected failure: highest error accepted by both information checks.
    candidates=[]
    for r in rows:
        if r['method']!='visible_bidirectional':continue
        for i,f in r['fits'].items():
            k=int(i)
            if r['truth_changed'][k] and r['base_assignment'][k]==r['truth_assignment'][k] and f['marginal_rank']==3:
                candidates.append(dict(scene=r['scene'],split=r['split'],observation=k,asset_id=r['truth_assignment'][k],estimated_delta=f['delta'],true_delta=r['truth_delta'][k],error=float(np.linalg.norm(np.array(f['delta'])-r['truth_delta'][k])),conditional_eigenvalues=f['conditional_eigenvalues'],marginal_eigenvalues=f['marginal_eigenvalues']))
    trace=max(candidates,key=lambda v:v['error']);(out/'false_confidence_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    # Oracle replacement diagnoses this outcome-selected example only.
    sid=trace['scene'];source=out/'cases' if trace['split']=='fresh' else Path('results/partial_surface_v1/cases');obs=load_input(source/f'{sid:04d}_input.npz');truth=dict(np.load(source/f'{sid:04d}_truth.npz'));base=manifest['base_predictions'][str(sid)];T=np.array(base['T']);A=np.array(base['prediction_assignment']);i=trace['observation']
    oracle=fit_motion(obs,truth['T'],A,i,cfg,'visible_bidirectional',stable_pose_information(obs,truth['T'],A,np.ones(len(A))))
    from scipy.spatial.transform import Rotation
    stable_r=np.array([np.median(np.abs(signed_box_distance(transform(q['points'],T),obs['bounds'][j]))) for q,j in zip(obs['query'],A)])
    stable_w=1/(1+(stable_r/.10)**4)
    support_trace=[dict(observation=k,asset_id=int(A[k]),true_changed=bool(truth['changed'][k]),median_sdf_m=float(stable_r[k]),stable_weight=float(stable_w[k])) for k in range(len(A))]
    oracle_diagnostic=dict(status='Posthoc outcome-selected single-case oracle diagnostic; true pose evaluator-only, not deployable method',scene=sid,pose_translation_error_m=float(np.linalg.norm(T[:3,3]-truth['T'][:3,3])),pose_rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(T[:3,:3]@truth['T'][:3,:3].T).magnitude())),estimated_pose_motion_error_m=trace['error'],true_pose_motion_error_m=float(np.linalg.norm(np.array(oracle['delta'])-truth['delta'][i])),true_pose_delta=oracle['delta'],stable_support_trace=support_trace)
    (out/'oracle_pose_failure_diagnostic.json').write_text(json.dumps(oracle_diagnostic,indent=2),encoding='utf-8')
    analysis=dict(summary=summary,fresh_paired_intervals=intervals,audit_checks=checks,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mainline_audit=dict(progress='Local motion estimation upgraded and diagnosed on frozen v1 pipeline; no pose/change classification improvement claimed.',engineering_gain='Visibility filtering eliminates boundary fits and reduces tail error relative to ungated reverse fitting.',paper_gain_status='Not established against one-sided control: fresh paired intervals include zero; information-rank check accepts substantial errors.',research_problem='Local full-rank information does not certify finite motion accuracy under pose/model bias and changing visibility. Need nuisance-robust finite-hypothesis evidence, with stronger existing methods.',next_gate='Do not add more local-fit terms or inflate these seeds; formalize camera/model nuisance hypotheses and matched strong controls before next algorithm branch.'))
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['split','method']+list(summary['fresh']['one_sided']));w.writeheader()
        for s,d in summary.items():
            for m,v in d.items():w.writerow(dict(split=s,method=m,**v))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');colours=['#888888','#E69F00','#0072B2'];names=['One-sided','Bidirectional','Visible bidirectional']
    xmax=max(v['maximum_vector_error_m'] for s in summary.values() for v in s.values())*1.05
    for ax,split in zip(axes,('reused','fresh')):
        for m,c,name in zip(cfg['methods'],colours,names):
            x=np.sort([v['error'] for r in rows if r['method']==m and r['split']==split for v in entries(r)]);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name,color=c)
        ax.axvline(.1,color='black',ls=':',lw=1);ax.set_xlim(0,xmax);ax.set_ylim(0,1.02);ax.set_xlabel('3D motion vector error (m)');ax.set_ylabel('Cumulative fraction');ax.set_title(('128 reused' if split=='reused' else '64 fresh')+' procedural scenes');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed v1 poses / associations / change decisions; local motion fits only\nCDF conditioned on correctly detected changed assets; missed changes reported separately.',fontsize=10)
    fig.savefig(out/'motion_errors.png',dpi=180);fig.savefig(out/'motion_errors.pdf');plt.close(fig)
    print(json.dumps(analysis,indent=2));print('False confidence trace:',json.dumps(trace))


if __name__=='__main__':main()
