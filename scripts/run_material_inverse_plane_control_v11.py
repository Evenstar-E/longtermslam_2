"""Fixed-candidate, fixed-identity-graph test of standard normal compatibility."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.run_material_surface_patch_registration_v9 import setup
from inspection_decisions.material_oriented_surface_control_v10 import orient_banks,OrientedScore,inverse
from inspection_decisions.material_surface_patch_registration_v9 import SurfaceScore
from inspection_decisions.material_inverse_plane_control_v11 import resample_banks,project as inverse_project
from inspection_decisions import material_oriented_surface_control_v10 as oriented
original_setup=setup
def setup(cfg,v):
    data=original_setup(cfg,v);resample_banks(data[7],data[1],cfg);oriented.project=inverse_project;return data
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['raw_points','raw_patch','normal_points','normal_patch'];PAIRS=[(0,1),(0,2),(1,2)]

def configuration():
    cp=Path('configs/material_oriented_surface_control_v10.json');c=json.loads(cp.read_text(encoding='utf-8'));cfg=json.loads(Path(c['surface_config']).read_text(encoding='utf-8'));cfg.update(c);v11=json.loads(Path('configs/material_inverse_plane_control_v11.json').read_text(encoding='utf-8'));cfg.update(v11);return cfg

def job(cfg,v):
    torch.set_num_threads(1);root=Path(cfg['output']);parent=Path(cfg['parent_root']);start=time.perf_counter();_,frames,_,bc,ep,groups,views,banks,edges=setup(cfg,v);orient_banks(banks,frames,cfg)
    np.savez_compressed(root/f'{v}_patches.npz',**{f'{e}_{i}_{key}':a[key] for e,row in enumerate(banks) for i,a in enumerate(row) for key in ['patch_xyz','patch_rgb','patch_uv','patch_frame','patch_normal','patch_normal_valid','patch_center_index']})
    with np.load(parent/f'{v}_pools.npz') as packed:pools=dict(packed)
    with np.load(parent/f'{v}_cache.npz') as packed:previous=dict(packed)
    with np.load(parent/f'{v}_search_graph.npz') as packed:pg=dict(packed)
    with np.load(Path(cfg['coupled_parent_root'])/f'{v}_cache.npz') as packed:coupled=dict(packed)
    lookup={tuple(t):q for q,t in enumerate(previous['triples'])};clook={tuple(t):q for q,t in enumerate(coupled['triples'])};cache=[];actual=0;full=0;agreement=0.
    @lru_cache(maxsize=256)
    def scorer(pe,i,j):
        e,f=PAIRS[pe];return OrientedScore(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,cfg)
    @lru_cache(maxsize=512)
    def pair_values(pe,i,j):
        T=pools['transforms'][pe,i,j,:pools['counts'][pe,i,j]];return scorer(pe,i,j).score(T)
    # Real public-input equivalence of the ungated implementation, no GT.
    A=pools['transforms'][0,0,0,:pools['counts'][0,0,0]];legacy=SurfaceScore(banks[0][0],banks[1][0],views[0][0],views[1][0],0,0,cfg).score(A)[:,2:];agreement=0.0
    for q,(i,j,k) in enumerate(previous['triples']):
        i,j,k=map(int,(i,j,k));A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]];T1=np.repeat(A,len(B),0);T2=np.tile(B,(len(A),1,1));relative=T2@inverse(T1);lower=(pair_values(0,i,j)[:,None]+pair_values(1,i,k)[None]).reshape(-1,4);seeds=np.unique(lower.argmin(0));value=lower[seeds]+.5*scorer(2,j,k).score(relative[seeds]);actual+=len(seeds);best=value.argmin(0);cc=value[best,np.arange(4)].copy();indices=seeds[best].copy()
        order=np.argsort(lower.min(1),kind='stable')
        for pos in range(0,len(order),32):
            hh=order[pos:pos+32];hh=hh[(lower[hh]<=cc[None]+1e-12).any(1)]
            if not len(hh):continue
            value=lower[hh]+.5*scorer(2,j,k).score(relative[hh]);actual+=len(hh)
            for m in range(4):
                n=int(value[:,m].argmin());val=value[n,m]
                if val<cc[m] or (val==cc[m] and hh[n]<indices[m]):cc[m]=val;indices[m]=hh[n]
        TT=np.array([[T1[h],T2[h]] for h in indices]);origin=np.zeros(4,int)
        if (i,j,k) in clook:
            U=coupled['transforms'][clook[i,j,k]];value=scorer(0,i,j).score(U[:,0])+scorer(1,i,k).score(U[:,1])+.5*scorer(2,j,k).score(U[:,1]@inverse(U[:,0]));h=value.argmin(0)
            for m,n in enumerate(h):
                if value[n,m]<cc[m]:cc[m]=value[n,m];TT[m]=U[n];origin[m]=n+1
        cache.append((cc,TT,origin));full+=len(T1)
        if q%8==7:save(root/f'{v}_progress.json',dict(triples=q+1,total=len(previous['triples']),elapsed_s=time.perf_counter()-start,full_pose_pairs=full,actual_normal_pair_scores=actual))
    costs=np.array([r[0] for r in cache]);values=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(4)) for s in pg['states']]);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=previous['triples'],costs=costs,transforms=np.array([r[1] for r in cache]),origins=np.array([r[2] for r in cache]));np.savez_compressed(root/f'{v}_search_graph.npz',states=pg['states'],values=values,best=best)
    for col,m in enumerate(METHODS):
        h=int(best[col]);state=pg['states'][h];TT=np.array([cache[lookup[i,int(j),int(k)]][1][col] for i,(j,k) in enumerate(state)]);cc=np.array([cache[lookup[i,int(j),int(k)]][0][col] for i,(j,k) in enumerate(state)]);np.savez_compressed(root/f'{v}_{m}.npz',target_slots=state,transforms=TT,objectives=cc,search_state=h,triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,elapsed_s=time.perf_counter()-start,triples=len(cache),states=len(pg['states']),full_pose_pairs=full,actual_normal_pair_scores=actual,raw_control_max_difference=None,representation_changed=True,private_labels_read=False,fixed_candidates_and_states=True,no_new_poses=True,valid_patch_normals=sum(int(a['patch_normal_valid'].sum()) for row in banks for a in row)));print('ORIENTED DONE',v,round(time.perf_counter()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();cfg=configuration()
    if args.profile:
        _,frames,_,_,_,_,views,banks,_=setup(cfg,'rich');orient_banks(banks,frames,cfg);s=OrientedScore(banks[0][0],banks[1][0],views[0][0],views[1][0],0,0,cfg)
        with np.load(Path(cfg['parent_root'])/'rich_pools.npz') as packed:T=packed['transforms'][0,0,0,:packed['counts'][0,0,0]]
        st=time.perf_counter();new=s.score(T);print(json.dumps(dict(public_only=True,poses=len(T),seconds=time.perf_counter()-st,finite=bool(np.isfinite(new).all()))));return
    parent=Path(cfg['parent_root']);index=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in index[group].items():assert sha(p)==h,p
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);paths=[__file__,'configs/material_oriented_surface_control_v10.json','configs/material_inverse_plane_control_v11.json','inspection_decisions/material_inverse_plane_control_v11.py','inspection_decisions/material_oriented_surface_control_v10.py','scripts/check_material_inverse_plane_control_v11.py','scripts/prepare_material_inverse_plane_control_v11.py',cfg['surface_config']];sources={p:sha(p) for p in paths};start=time.perf_counter();save(root/'frozen_protocol.json',dict(config=cfg,sources=sources,methods=METHODS,parent_completion_sha=sha(parent/'completion_integrity.json'),scope='Fixed V9 pool (including original V8 coupled candidates), exact same V9 finite identity states, no new continuous poses. Inverse-depth plane support repair, all four scores recomputed on common new patches, not novel theory. Outcome-informed single TRAIN correction, not independent confirmation.'))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')}));print('ORIENTED SEALED',flush=True)

if __name__=='__main__':main()
