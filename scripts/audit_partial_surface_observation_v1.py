from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.partial_surface_observation_v1 import capture,prepare,arrays,full_energy,surface_seeds
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_partial_surface_observation_v1 import CFG,solver,cases
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    cfg=read(CFG);out=Path(cfg['output']);f=read(out/'frozen_contract.json');a=read(out/'analysis.json');checks=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in f['sources'].items():ck(digest(p)==sha)
    for sid,case in enumerate(cases(cfg)):
        root=out/'scenes'/f'{sid:04d}';saved_raw=read(out/f'{sid:04d}_observed.json');backends=read(out/f'{sid:04d}_backends.json')
        # Real API reacquisition and full feature/reference/backend reconstruction, not only replay from pair tables.
        ep=Episode(root,2,1);h=[ep.act('retrieve_history',0),ep.act('retrieve_history',1)];q=[ep.preview(),ep.act('observe_current',1)]
        ck(canonical_bytes(capture(h,q))==canonical_bytes(saved_raw['raw']))
        ck(canonical_bytes(prepare(saved_raw['raw'],cfg))==canonical_bytes(backends))
        catalog=read(root/'catalog.json');ck(len(saved_raw['events'])==4)
        for event in saved_raw['events']:
            token=catalog[event['epoch']][event['view_id']]['token'];ck(digest(root/'payloads'/f'{token}.npz')==event['payload_sha256'])
        cost=ep.finish();saved_cost=read(out/f'{sid:04d}_cost.json')
        for name in ['history_views','additional_current_views','payload_bytes','travel_m','returned','closed']:
            ck(abs(cost[name]-saved_cost[name])<1e-9)
        for method in cfg['methods']:
            r=read(out/f'{sid:04d}_{method}.json');row=next(x for x in a['rows'] if x['scene']==sid and x['method']==method)
            ck(canonical_bytes(solver(backends,method,cfg))==canonical_bytes(r))
            ck(canonical_bytes(evaluate(root,r['prediction']))==canonical_bytes(row['metric']))
            ck(canonical_bytes(saved_cost)==canonical_bytes(row['cost']))
            ck(np.max(abs(np.array(r['best']['translation'])))<=cfg['public_session_translation_bound_m']+1e-8)
            for branch in r['branches']:
                for pairs in branch['matches']:ck(len({i for i,j in pairs})==len(pairs) and len({j for i,j in pairs})==len(pairs))
                if method!='legacy_robust':ck(np.max(np.diff(branch['energy_trace']),initial=0)<1e-6)
                if method=='surface_directional':
                    e=full_energy(arrays(backends['surface']),branch['matches'],np.array(branch['translation']),np.array(branch['delta']),cfg)
                    ck(abs(e-branch['score'])<1e-7)
    intervention=[]
    for family in range(cfg['paired_families']):
        roots=[out/'scenes'/f'{2*family+k:04d}' for k in [0,1]];truth=[read(r/'evaluation/truth.json') for r in roots]
        for key in ['historical_vertices','current_vertices','delta','changed','poses','realized_nuisance','injected_session_translation']:
            ck(canonical_bytes(truth[0][key])==canonical_bytes(truth[1][key]))
        changed_pixels=0
        for epoch in ['history','current']:
            for view in range(12):
                token=f'{epoch}_{view:02d}'
                with np.load(roots[0]/'payloads'/f'{token}.npz') as x,np.load(roots[1]/'payloads'/f'{token}.npz') as y:
                    ck(np.array_equal(x['nominal_T'],y['nominal_T']))
                    same=x['depth']==y['depth'];ck(np.array_equal(x['rgb'][same],y['rgb'][same]))
                    if epoch=='history':ck(bool(same.all()) and np.array_equal(x['rgb'],y['rgb']))
                    elif view in [0,1]:changed_pixels+=int((~same).sum())
        ck(changed_pixels>0);intervention.append(dict(family=family,selected_current_changed_pixels=changed_pixels))
    rng=np.random.default_rng(2318026);ix=rng.integers(0,cfg['paired_families'],(5000,cfg['paired_families']));contrasts={}
    def losses(method,condition):return np.array([r['metric']['task_loss'] for r in a['rows'] if r['method']==method and r['condition']==condition])
    for method in cfg['methods'][1:]:
        d=losses('surface_directional','occluded')-losses(method,'occluded')
        clear=losses('surface_directional','clear')-losses(method,'clear');interaction=d-clear
        contrasts[method]=dict(mean_occluded_loss_difference=float(d.mean()),exploratory_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),
            occluded_differences=d.tolist(),clear_differences=clear.tolist(),mean_occlusion_interaction=float(interaction.mean()),
            interaction_ci95=np.quantile(interaction[ix].mean(1),[.025,.975]).tolist(),improved=int(sum(d<0)),worse=int(sum(d>0)),tied=int(sum(d==0)))
    save(out/'audit.json',dict(status='PASS',checks=checks,raw_API_replays=20,backend_reconstructions=20,solver_replays=100,endpoint_reevaluations=100,
        paired_visibility_interventions=intervention,contrasts=contrasts,initial_freeze_unchanged=True,exclusions=[],
        inference='8 paired worlds for primary occlusion contrast; clear/occluded NOT independent; unadjusted exploratory intervals'))
    print('AUDIT PASS',checks,contrasts,flush=True)


if __name__=='__main__':main()
