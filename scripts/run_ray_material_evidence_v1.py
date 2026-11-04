"""Label-blind shared-pool model experiment. Never opens evaluation directory."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair

def perturb(T,anchor,axis,amount):
    out=T.copy();p=T[:3,:3]@anchor+T[:3,3]
    if axis<3:
        v=np.zeros(3);v[axis]=amount;out[:3,:3]=Rotation.from_rotvec(v).as_matrix()@T[:3,:3];out[:3,3]=p-out[:3,:3]@anchor
    else:out[axis-3,3]+=amount
    return out

def main():
    config=Path('configs/ray_material_evidence_v1.json');cfg=read(config);root=Path(cfg['public_root']);public=read(root/'public.json');out=Path('results/ray_material_evidence_v1');out.mkdir(exist_ok=False)
    sources=[Path(__file__),config,Path('inspection_decisions/ray_material_evidence_v1.py')]
    inputs=[root/'public.json']+[p for p in (root/'public').rglob('*') if p.is_file()]+list(Path(cfg['proposal_root']).glob('pair_*/predictions.json'))
    oldseal=read(Path(cfg['proposal_root'])/'prediction_seal.json')
    for name,h in oldseal['sha256'].items():assert sha(Path(cfg['proposal_root'])/name)==h
    save(out/'start_manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},GT_pose_read=False,scope=cfg['scope']))
    frames={r['stem']:Frame(root,r['stem'],cfg) for r in public['clouds']};started=time.perf_counter()
    for k,pair in enumerate(public['pairs']):
        name=f'pair_{k:02d}';old=read(Path(cfg['proposal_root'])/name/'predictions.json');anchor=np.array(old['historical_observed_anchor_m']);model=RayPair(frames[pair['history_stem']],frames[pair['current_stem']],anchor,cfg);pool=[];cache={}
        def add(T,source):
            key=np.round(T,11).tobytes()
            if key in cache:return cache[key]
            scores,diagnosis=model.score(T);r=dict(T_current_from_history=T.tolist(),source=source,scores=scores,diagnosis=diagnosis);pool.append(r);cache[key]=len(pool)-1
            assert abs(scores['beta_material']-scores['ordinary_exact_beta_equivalent'])<1e-12
            return len(pool)-1
        starts=[add(np.array(r['T_current_from_history']),r['method']) for r in old['methods']]
        starts.append(add(np.array(old['candidate']['poses'][0]),'old_bounded_candidate'))
        starts=list(dict.fromkeys(starts));initial=list(starts);optimizer_outputs=[]
        for method in cfg['refinement_methods']:
            for start in starts:
                best=start
                for rot,trans in zip(cfg['refinement_rotation_steps_deg'],cfg['refinement_translation_steps_m']):
                    for _ in range(cfg['refinement_rounds_per_scale']):
                        T=np.array(pool[best]['T_current_from_history']);neighbors=[best]
                        for axis in range(6):
                            for sign in [-1,1]:neighbors.append(add(perturb(T,anchor,axis,sign*(np.deg2rad(rot) if axis<3 else trans)),method+'_coordinate_search'))
                        best=min(neighbors,key=lambda j:pool[j]['scores'][method])
                optimizer_outputs.append(dict(method=method,start=start,best=best))
        center=np.eye(4);center[:3,3]=model.c.all_xyz.mean(0)-model.h.all_xyz.mean(0);escape=[]
        for axis in range(3):
            for sign in [-1,1]:escape.append(add(perturb(center,anchor,axis+3,sign*.75),'unknown_escape_control'))
        choices={method:min(range(len(pool)),key=lambda j:pool[j]['scores'][method]) for method in cfg['methods']}
        restricted={method:min(initial,key=lambda j:pool[j]['scores'][method]) for method in cfg['methods']}
        dest=out/name;dest.mkdir();save(dest/'predictions.json',dict(pair=pair,anchor=anchor.tolist(),pool=pool,choices=choices,original_proposals_only=restricted,initial_indices=initial,escape_indices=escape,optimizer_outputs=optimizer_outputs,unique_full_score_evaluations=model.calls,GT_pose_read=False))
        print('RAY_MODEL_DONE',name,'poses',len(pool),'elapsed',time.perf_counter()-started,flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_pose_read=False,elapsed_s=time.perf_counter()-started,complete=True))

if __name__=='__main__':main()
