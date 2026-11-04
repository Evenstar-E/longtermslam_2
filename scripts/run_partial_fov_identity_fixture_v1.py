"""Actual RGB-D partial field-of-view fixture; no performance/novelty claim."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from comparative_evidence.scene import vertices,halfspaces,render
from comparative_evidence.interface import Episode
from comparative_evidence.estimation_v1 import reference_map,world_components
from scripts.partial_surface_scene import camera
from inspection_decisions.projective_visibility_v1 import capture,decode
from inspection_decisions.se3_visibility_v1 import register
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_evidence_value_pilot_v1 import read,save

CFG='configs/partial_fov_identity_fixture_v1.json'


def main():
    cfg=read(CFG);reg=read('configs/se3_visibility_v1.json');reg.update({k:cfg[k] for k in ['se3_translation_bound_m','se3_rotation_bound_deg']})
    sensor=read('configs/comparative_evidence_interface_v1.json');sensor['factors']={};out=Path(cfg['output']);assert not out.exists();out.mkdir(parents=True)
    paths=[CFG,__file__,'inspection_decisions/se3_visibility_v1.py','comparative_evidence/scene.py','comparative_evidence/interface.py']
    sources={p:digest(p) for p in paths}
    save(out/'fixture_contract.json',dict(status='frozen before fixture generation; designed after main pilot mechanism audit',config=cfg,registration_config=reg,sources=sources,
        claim='only acceptance of actual competing identity mappings in acquired partial RGBD, no estimator superiority',
        geometry='aligned repetitive facade panels, small geometric jitter; history wide view,current closer static view; no invented feature candidates',
        prior='3m local-map origin envelope allows adjacent repetitive columns; design choice, not measured GNSS statistics'))
    rows=[];checks=0
    for world in range(cfg['worlds']):
        root=out/'scenes'/f'{world:04d}';(root/'payloads').mkdir(parents=True);(root/'evaluation').mkdir()
        rng=np.random.default_rng(cfg['seed_start']+world)
        centres=np.array([[(j%4-1.5)*1.65,(j//4-.5)*1.65,5.5] for j in range(8)])+rng.normal(0,cfg['layout_jitter_m'],(8,3))
        shapes=[[p*.75+c for p in vertices('hex_prism')] for c in centres];parts=[halfspaces(s) for s in shapes];groups=np.zeros(8,int)
        G=np.eye(4);G[:3,:3]=Rotation.from_euler('xyz',[1.,-1.,7.],degrees=True).as_matrix();G[:3,3]=[.45,-.1,.2]
        catalog=dict(sensor=sensor,history=[],current=[],edges=[],error_model=dict(map_registration_prior=dict(translation_infinity_bound_m=3.,rotation_angle_bound_deg=15.)))
        for epoch,radius in [('history',cfg['history_radius_m']),('current',cfg['current_radius_m'])]:
            T=camera(np.array([0.,0.,5.5-radius]),np.array([0.,0.,5.5]));rgb,depth,owner=render(parts,T,groups,sensor,rng)
            nominal=G@T if epoch=='current' else T;token=f'{epoch}_00'
            np.savez_compressed(root/'payloads'/f'{token}.npz',rgb=rgb,depth=depth,nominal_T=nominal)
            np.savez_compressed(root/'evaluation'/f'{token}_owner.npz',owner=owner)
            catalog[epoch].append(dict(id=0,token=token,nominal_T=nominal.tolist(),nuisance_group=epoch))
        save(root/'catalog.json',catalog);save(root/'evaluation/truth.json',dict(session_G=G.tolist(),centres=centres.tolist(),scope='static partial-FOV input acceptance only'))
        ep=Episode(root,1,0);hs=[ep.act('retrieve_history',0)];qs=[ep.preview()];raw=capture(hs,qs,sensor)
        save(out/f'{world:04d}_observed.json',dict(raw=raw,events=ep.state()['events']))
        result=register(raw,reg);save(out/f'{world:04d}_registration.json',result);cost=ep.finish();save(out/f'{world:04d}_cost.json',cost)
        # Evaluation owner images used only after raw inputs and registration outputs persist.
        refs=reference_map([decode(o) for o in raw['history']]);cs=world_components(decode(raw['current'][0]))
        with np.load(root/'evaluation/history_00_owner.npz') as z:past=z['owner'].ravel()
        with np.load(root/'evaluation/current_00_owner.npz') as z:present=z['owner'].ravel()
        owner_ref=[int(np.bincount(past[r['anchor_pixels']][past[r['anchor_pixels']]>=0],minlength=8).argmax()) for r in refs]
        owner_cur=[int(np.bincount(present[c['pixels']][present[c['pixels']]>=0],minlength=8).argmax()) for c in cs]
        hypotheses=[]
        for r in result['retained']:
            pairs=r['matches'][0];hypotheses.append(dict(matches=pairs,score=r['score'],physical_pairs=[(owner_cur[i],owner_ref[j]) for i,j in pairs],
                correct=sum(owner_cur[i]==owner_ref[j] for i,j in pairs)))
        distinct=len({str(r['matches']) for r in result['retained']})
        assert canonical_bytes(register(raw,reg))==canonical_bytes(result);checks+=1
        assert len(cs)<len(refs);checks+=1
        rows.append(dict(world=world,historical_references=len(refs),current_components=len(cs),retained_poses=len(result['retained']),retained_actual_assignments=distinct,
            hypotheses=hypotheses,returned=cost['returned']))
    assert all(digest(p)==sha for p,sha in sources.items());checks+=len(sources)
    save(out/'analysis.json',dict(status='input acceptance only',rows=rows,checks=checks,raw_RGBD_frames=2*cfg['worlds'],API_paths=cfg['worlds'],registration_replays=cfg['worlds'],
        worlds_with_multiple_actual_assignments=sum(r['retained_actual_assignments']>1 for r in rows),task_success_claim=False,innovation_claim=False,exclusions=[]))
    print('PARTIAL FOV FIXTURE',rows,'checks',checks,flush=True)


if __name__=='__main__':main()
