from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.correspondence_motion_objective_v1 import energy,frames_from
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_correspondence_motion_objective_v1 import CFG,solver
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    cfg=read(CFG);out=Path(cfg['output']);f=read(out/'frozen_contract.json');a=read(out/'analysis.json');checks=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in f['sources'].items():ck(digest(p)==sha)
    for sid in range(cfg['scenes']):
        root=out/'scenes'/f'{sid:04d}';observed=read(out/f'{sid:04d}_observed.json');data=observed['evidence'];catalog=read(root/'catalog.json')
        ck(len(observed['events'])==4)
        for e in observed['events']:
            token=catalog[e['epoch']][e['view_id']]['token'];ck(digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'])
        for method in cfg['methods']:
            r=read(out/f'{sid:04d}_{method}.json');replay=solver(data,method,cfg)
            ck(canonical_bytes(r)==canonical_bytes(replay))
            row=next(v for v in a['rows'] if v['scene']==sid and v['method']==method)
            ck(canonical_bytes(evaluate(root,r['prediction']))==canonical_bytes(row['metric']))
            ck(canonical_bytes(read(out/f'{sid:04d}_cost.json'))==canonical_bytes(row['cost']))
            ck(np.max(abs(np.array(r['best']['translation'])))<=cfg['public_session_translation_bound_m']+1e-9)
            for branch in r['branches']:
                for pairs in branch['matches']:ck(len({j for i,j in pairs})==len(pairs) and len({i for i,j in pairs})==len(pairs))
                if method in ('shared_motion','shared_no_amplitude'):
                    ck(np.max(np.diff(branch['energy_trace']),initial=0)<1e-7)
                    recomputed=energy(frames_from(data),branch['matches'],np.array(branch['translation']),np.array(branch['delta']),cfg,method=='shared_no_amplitude')
                    ck(abs(recomputed-branch['score'])<1e-7)
    rng=np.random.default_rng(2316026);k=cfg['scenes']//len(cfg['regimes'])
    ix=np.concatenate([rng.integers(i*k,(i+1)*k,(5000,k)) for i in range(len(cfg['regimes']))],axis=1)
    contrasts={}
    candidate=[r for r in a['rows'] if r['method']=='shared_motion']
    for method in cfg['methods'][1:]:
        control=[r for r in a['rows'] if r['method']==method]
        d=np.array([c['metric']['task_loss']-b['metric']['task_loss'] for c,b in zip(candidate,control)])
        contrasts[method]=dict(mean_loss_difference=float(d.mean()),exploratory_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),
            scene_loss_differences=d.tolist(),improved=int(sum(d<0)),worse=int(sum(d>0)),tied=int(sum(d==0)))
    save(out/'audit.json',dict(status='PASS',checks=checks,solver_replays=len(a['rows']),endpoint_reevaluations=len(a['rows']),
        contrasts=contrasts,initial_freeze_unchanged=True,exclusions=[],
        inference='16 condition-stratified paired families; unadjusted exploratory intervals, NOT confirmation'))
    print('AUDIT PASS',checks,contrasts,flush=True)


if __name__=='__main__':main()
