from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture,prepare
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_projective_visibility_v1 import CFG,solver
from scripts.run_partial_surface_observation_v1 import cases
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    cfg=read(CFG);out=Path(cfg['output']);f=read(out/'frozen_contract.json');a=read(out/'analysis.json');checks=0;fits=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in f['sources'].items():ck(digest(p)==sha)
    for sid,case in enumerate(cases(cfg)):
        root=out/'scenes'/f'{sid:04d}';raw=read(out/f'{sid:04d}_observed.json');backend=read(out/f'{sid:04d}_backend.json')
        ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
        ck(canonical_bytes(capture(h,q,ep.metadata()['sensor']))==canonical_bytes(raw['raw']))
        ck(canonical_bytes(prepare(raw['raw'],cfg))==canonical_bytes(backend))
        ck(len(raw['events'])==4);catalog=read(root/'catalog.json');cost=ep.finish();saved_cost=read(out/f'{sid:04d}_cost.json')
        for e in raw['events']:
            token=catalog[e['epoch']][e['view_id']]['token'];ck(digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'])
        for name in ['history_views','additional_current_views','payload_bytes','travel_m','returned','closed']:ck(abs(cost[name]-saved_cost[name])<1e-9)
        initial_matches=backend['initial']['best']['matches']
        for diag in backend['diagnostics']:
            for method,records in diag['fits'].items():
                use=records if method=='independent_censored' else records[:1]
                for fit in use:
                    fits+=1;ck(np.max(np.diff(fit['trace']),initial=0)<1e-9)
                    ck(fit['loss']<=fit['initial_loss']+1e-9 and fit['evaluations']<=61)
                    ck(np.max(abs(np.array(fit['d'])-diag['start']))<=cfg['local_search_bound_m']+1e-9)
        for method in cfg['methods']:
            r=read(out/f'{sid:04d}_{method}.json');row=next(x for x in a['rows'] if x['scene']==sid and x['method']==method)
            ck(canonical_bytes(solver(backend,method,cfg))==canonical_bytes(r));ck(canonical_bytes(evaluate(root,r['prediction']))==canonical_bytes(row['metric']))
            ck(canonical_bytes(saved_cost)==canonical_bytes(row['cost']))
            for branch in r['branches']:
                for frame,pairs in enumerate(branch['matches']):
                    ck(len({i for i,j in pairs})==len(pairs) and len({j for i,j in pairs})==len(pairs))
                    if method not in ['legacy_shared','legacy_robust']:ck(set(map(tuple,pairs))<=set(map(tuple,initial_matches[frame])))
                if method!='legacy_robust':ck(np.max(np.diff(branch['energy_trace']),initial=0)<1e-6)
        print('audited scene',sid,flush=True)
    interventions=[]
    for family in range(cfg['paired_families']):
        roots=[out/'scenes'/f'{family*2+k:04d}' for k in [0,1]];ts=[read(r/'evaluation/truth.json') for r in roots]
        for key in ['historical_vertices','current_vertices','changed','delta','poses','realized_nuisance','injected_session_translation']:
            ck(canonical_bytes(ts[0][key])==canonical_bytes(ts[1][key]))
        changed=0
        for epoch in ['history','current']:
            for v in range(12):
                with np.load(roots[0]/'payloads'/f'{epoch}_{v:02d}.npz') as x,np.load(roots[1]/'payloads'/f'{epoch}_{v:02d}.npz') as y:
                    ck(np.array_equal(x['nominal_T'],y['nominal_T']));same=x['depth']==y['depth'];ck(np.array_equal(x['rgb'][same],y['rgb'][same]))
                    if epoch=='history':ck(bool(same.all()) and np.array_equal(x['rgb'],y['rgb']))
                    elif v in [0,1]:changed+=int((~same).sum())
        ck(changed>0);interventions.append(dict(family=family,selected_changed_pixels=changed))
    rng=np.random.default_rng(2320026);ix=rng.integers(0,cfg['paired_families'],(5000,cfg['paired_families']));contrasts={}
    def losses(m,c):return np.array([r['metric']['task_loss'] for r in a['rows'] if r['method']==m and r['condition']==c])
    for method in cfg['methods'][1:]:
        d=losses('joint_censored','occluded')-losses(method,'occluded');clear=losses('joint_censored','clear')-losses(method,'clear');interaction=d-clear
        contrasts[method]=dict(mean_occluded_loss_difference=float(d.mean()),exploratory_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),
            occluded_differences=d.tolist(),clear_differences=clear.tolist(),mean_interaction=float(interaction.mean()),
            interaction_ci95=np.quantile(interaction[ix].mean(1),[.025,.975]).tolist(),improved=int(sum(d<0)),worse=int(sum(d>0)),tied=int(sum(d==0)))
    save(out/'audit.json',dict(status='PASS',checks=checks,raw_API_replays=20,backend_reconstructions=20,local_search_replays=fits,
        final_solver_replays=120,endpoint_reevaluations=120,paired_visibility_interventions=interventions,contrasts=contrasts,
        initial_freeze_unchanged=True,exclusions=[],interpretation='8 paired world units primary; exploratory unadjusted intervals, not independent confirmation'))
    print('AUDIT PASS',checks,'localsearchreplays',fits,contrasts,flush=True)


if __name__=='__main__':main()
