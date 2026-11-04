from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import capture
from inspection_decisions.se3_visibility_v1 import run_methods
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_se3_visibility_v1 import CFG,cases,scene_root
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    cfg=read(CFG);out=Path(cfg['output']);frozen=read(out/'frozen_contract.json');a=read(out/'analysis.json');checks=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in frozen['sources'].items():ck(digest(p)==sha)
    for sid,case in enumerate(cases(cfg)):
        root=scene_root(out,case['family'],case['variant']);saved=read(out/f'{sid:04d}_observed.json');b=read(out/f'{sid:04d}_backend.json')
        ep=Episode(root,case['history_count'],2);hs=[ep.act('retrieve_history',v) for v in cfg['history_views'][:case['history_count']]]
        qs=[ep.preview()]+[ep.act('observe_current',v) for v in cfg['current_views'][1:]]
        ck(canonical_bytes(capture(hs,qs,ep.metadata()['sensor']))==canonical_bytes(saved['raw']))
        ck(len(saved['events'])==case['history_count']+3)
        catalog=read(root/'catalog.json')
        for e in saved['events']:
            token=catalog[e['epoch']][e['view_id']]['token'];ck(digest(root/'payloads'/f'{token}.npz')==e['payload_sha256'])
        replay=run_methods(saved['raw'],cfg);ck(canonical_bytes(replay)==canonical_bytes(b))
        cost=ep.finish();sc=read(out/f'{sid:04d}_cost.json')
        for name in ['history_views','additional_current_views','payload_bytes','travel_m','returned','closed']:ck(abs(cost[name]-sc[name])<1e-9)
        for method in cfg['methods']:
            r=read(out/f'{sid:04d}_{method}.json');row=next(rw for rw in a['rows'] if rw['scene']==sid and rw['method']==method)
            ck(canonical_bytes(r)==canonical_bytes(b['outputs'][method]))
            ck(canonical_bytes(evaluate(root,r['prediction']))==canonical_bytes(row['metric']))
            ck(r['prediction']['delta_frame']=='archive')
            G=np.array(r['initial_map_transform']);T=np.eye(4);T[:3,3]=r['best']['translation']
            ck(np.allclose(G@T,r['map_transform_current_from_archive'],atol=1e-12))
            ck(canonical_bytes(row['cost'])==canonical_bytes(sc))
            for branch in r['branches']:
                for pairs in branch['matches']:ck(len({i for i,j in pairs})==len(pairs) and len({j for i,j in pairs})==len(pairs))
                ck(np.max(np.diff(branch['energy_trace']),initial=0)<1e-6)
            for fit in r['projective_fits']:
                ck(np.max(np.diff(fit['fit']['trace']),initial=0)<1e-9 and fit['fit']['evaluations']<=61)
        print('audited case',sid,flush=True)
    paired=[]
    for family in range(cfg['world_families']):
        roots=[scene_root(out,family,v) for v in ['identity','se3']];ts=[read(r/'evaluation/truth.json') for r in roots]
        for key in ['historical_vertices','current_vertices','delta','changed','poses','realized_nuisance','base_current_nominal_poses']:
            ck(canonical_bytes(ts[0][key])==canonical_bytes(ts[1][key]))
        G=np.array(ts[1]['session_G'])
        for epoch in ['history','current']:
            for view in range(12):
                with np.load(roots[0]/'payloads'/f'{epoch}_{view:02d}.npz') as x,np.load(roots[1]/'payloads'/f'{epoch}_{view:02d}.npz') as y:
                    ck(np.array_equal(x['rgb'],y['rgb']) and np.array_equal(x['depth'],y['depth']))
                    ck(np.allclose(y['nominal_T'],G@x['nominal_T'] if epoch=='current' else x['nominal_T'],atol=1e-12))
        c=read(roots[1]/'catalog.json');radius=ts[1]['path_enclosing_radius_m'];target=np.array([0.,0.,5.5])
        for edge in c['edges']:
            ps=[(np.linalg.inv(G)@np.array(c['current'][idx]['nominal_T']))[:3,3] for idx in [edge['a'],edge['b']]]
            x,y=ps;d=y-x;u=np.clip((target-x)@d/(d@d),0,1);ck(np.linalg.norm(x+u*d-target)>radius)
            ck(abs(np.linalg.norm(d)-edge['distance_m'])<1e-9)
        paired.append(dict(family=family,scope='exact pixel pair; current nominal poses differ by SE3 only'))
    primary=[r for r in a['rows'] if r['condition']=='repeat_se3_h1'];families=sorted({r['family'] for r in primary});rng=np.random.default_rng(2322026);ix=rng.integers(0,len(families),(5000,len(families)));contrasts={}
    def losses(method):return np.array([next(r['metric']['task_loss'] for r in primary if r['family']==f and r['method']==method) for f in families])
    for method in cfg['methods'][1:]:
        d=losses('se3_late_visibility')-losses(method)
        contrasts[method]=dict(mean_loss_difference=float(d.mean()),exploratory_ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),differences=d.tolist(),
            improved=int(sum(d<0)),worse=int(sum(d>0)),tied=int(sum(d==0)))
    save(out/'audit.json',dict(status='PASS',checks=checks,raw_API_replays=30,full_backend_replays=30,method_output_replays=150,endpoint_reevaluations=150,
        exact_frame_pairs=paired,primary_contrasts=contrasts,initial_freeze_unchanged=True,exclusions=[],
        interpretation='6 repeated worlds primary, 12 total independent worlds; frame/history variants dependent, exploratory intervals'))
    print('AUDIT PASS',checks,contrasts,flush=True)


if __name__=='__main__':main()
