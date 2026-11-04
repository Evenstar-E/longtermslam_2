from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.correspondence_joint_v1 import solve
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_evidence_value_pilot_v1 import read,save

def main():
    cfg=read('configs/correspondence_joint_pilot_v1.json');out=Path(cfg['output']);f=read(out/'frozen_contract.json');a=read(out/'analysis.json');checks=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in f['sources'].items():ck(digest(p)==sha)
    paired=[];bad_cases=[]
    for sid in range(cfg['scenes']):
        root=out/'scenes'/f'{sid:04d}';observed=read(out/f'{sid:04d}_observed.json');data=observed['evidence'];catalog=read(root/'catalog.json')
        ck(len(observed['events'])==4)
        for e in observed['events']:
            token=catalog[e['epoch']][e['view_id']]['token'];ck(digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'])
        for method in cfg['methods']:
            saved=read(out/f'{sid:04d}_{method}.json');replay=solve(data,method,cfg)
            ck(canonical_bytes(saved)==canonical_bytes(replay))
            metric=evaluate(root,saved['prediction']);r=next(r for r in a['rows'] if r['scene']==sid and r['method']==method)
            ck(canonical_bytes(metric)==canonical_bytes(r['metric']))
            ck(np.max(abs(np.array(saved['best']['translation'])))<=cfg['public_session_translation_bound_m']+1e-9)
            if method!='nearest':
                for branch in saved['branches']:
                    for pairs in branch['matches']:ck(len({j for i,j in pairs})==len(pairs) and len({i for i,j in pairs})==len(pairs))
        c=next(r for r in a['rows'] if r['scene']==sid and r['method']=='joint_branches');b=next(r for r in a['rows'] if r['scene']==sid and r['method']=='robust_multistart')
        paired.append(dict(scene=sid,regime=c['regime'],loss_difference=c['metric']['task_loss']-b['metric']['task_loss'],qualified_difference=c['metric']['qualified_tp']-b['metric']['qualified_tp']))
        if c['metric']['identity_wrong_references']:
            cc=read(out/f'{sid:04d}_joint_branches.json');bb=read(out/f'{sid:04d}_robust_multistart.json')
            bad_cases.append(dict(scene=sid,regime=c['regime'],candidate_ownership=c['metric']['ownership_records'],control_ownership=b['metric']['ownership_records'],
                candidate_assignments=cc['best']['matches'],control_assignments=bb['best']['matches'],candidate_translation=cc['best']['translation'],control_translation=bb['best']['translation']))
    rng=np.random.default_rng(2314026);idx=np.concatenate([rng.integers(i*4,(i+1)*4,(5000,4)) for i in range(4)],axis=1)
    d=np.array([r['loss_difference'] for r in paired]);interval=np.quantile(d[idx].mean(1),[.025,.975])
    save(out/'audit.json',dict(status='PASS',checks=checks,solver_replays=64,endpoint_reevaluations=64,paired_rows=paired,
        mean_loss_difference=float(d.mean()),exploratory_ci95=interval.tolist(),identity_failure_scenes=bad_cases,
        interpretation='Pilot only,16pairedfamilies,4strata,unadjusted; no independent confirmation; no source/model tuning',
        initial_freeze_unchanged=True,exclusions=[]))
    print('AUDIT PASS',checks,'candidate-control mean loss',d.mean(),'CI',interval,'identityfailurescenes',[r['scene'] for r in bad_cases],flush=True)

if __name__=='__main__':main()
