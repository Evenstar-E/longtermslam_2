"""Full24 public-input continuous proposals with common final candidates."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,time
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.run_material_inverse_plane_control_v11 import configuration as previous_configuration,setup,PAIRS
from scripts.run_textured_material_comparison_v1 import save,sha
from inspection_decisions.material_oriented_surface_control_v10 import orient_banks,OrientedScore,inverse
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import (
    TripleScore,continuous_fit,joint_fit,unique_poses,propagation)

METHODS=['retained_points','retained_patch','propagated_points','propagated_patch',
         'continuous_points','continuous_patch','independent_material','shared_material']


def configuration():
    cfg=previous_configuration();cfg.update(json.loads(Path('configs/material_inverse_plane_pose_proposals_v12.json').read_text(encoding='utf-8')));return cfg


def job(cfg,v,profile=False):
    torch.set_num_threads(1);start=time.perf_counter();root=Path(cfg['output'])
    _,frames,_,_,_,_,views,banks,_=setup(cfg,v);orient_banks(banks,frames,cfg)
    def load(p):
        with np.load(p) as f:return dict(f)
    pool=load(Path(cfg['parent_root'])/f'{v}_pools.npz')
    old=load(Path(cfg['retained_root'])/f'{v}_cache.npz');graph=load(Path(cfg['retained_root'])/f'{v}_search_graph.npz')
    v9=load(Path(cfg['parent_root'])/f'{v}_cache.npz');v8=load(Path(cfg['coupled_parent_root'])/f'{v}_cache.npz')
    look9={tuple(t):h for h,t in enumerate(v9['triples'])};look8={tuple(t):h for h,t in enumerate(v8['triples'])}
    @lru_cache(maxsize=256)
    def scorer(pe,i,j):
        e,f=PAIRS[pe];return OrientedScore(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,cfg)
    @lru_cache(maxsize=512)
    def pair(pe,i,j):
        T=pool['transforms'][pe,i,j,:pool['counts'][pe,i,j]];return T,scorer(pe,i,j).score(T)
    def modes(pe,i,j):
        T,c=pair(pe,i,j);indices=np.unique(np.argsort(c,axis=0,kind='stable')[:cfg['proposal_modes_per_score']].ravel());return T[indices]
    rows=[];allproposals={};fitrecords={};actual=0
    counters=dict(composed_candidates=0,ordinary_fits=0,ordinary_improved=0,independent_fits=0,shared_fits=0,joint_improved=0)
    for q,triple in enumerate(old['triples'][:1] if profile else old['triples']):
        i,j,k=map(int,triple);key=i,j,k;sc=TripleScore([scorer(0,i,j),scorer(1,i,k),scorer(2,j,k)],cfg)
        retained=list(old['transforms'][q])
        if key in look9:retained+=list(v9['transforms'][look9[key]])
        if key in look8:retained+=list(v8['transforms'][look8[key]])
        retained=unique_poses(retained);compose=propagation(modes(0,i,j),modes(1,i,k),modes(2,j,k))
        U=unique_poses(list(retained)+list(compose));stage=np.array([0 if any(np.max(abs(T-R))<1e-9 for R in retained) else 1 for T in U]);values=sc.score(U)
        counters['composed_candidates']+=int((stage==1).sum());cc=np.zeros(8);TT=np.zeros((8,2,4,4));origin=np.zeros(8,int)
        cc[:2]=old['costs'][q,2:4];TT[:2]=old['transforms'][q,2:4];origin[:2]=-1
        for m in range(2):
            h=int(values[:,m].argmin());cc[m+2]=min(cc[m],values[h,m]);TT[m+2]=TT[m] if cc[m]<=values[h,m] else U[h];origin[m+2]=-1 if cc[m]<=values[h,m] else h
        # Both ordinary point/patch directions start from the SAME best point/patch union.
        starts=np.unique(values[:,:2].argmin(0))[:cfg['continuous_starts']];new=[];paths=[]
        for h in starts:
            for col in [2,3]:
                pose=[]
                for e in range(2):
                    t,path=continuous_fit(sc.scorers[e],U[h,e],cfg,col);pose.append(t);paths.append((h,col,e,path));counters['ordinary_fits']+=1;counters['ordinary_improved']+=int(path[-1]<path[0]-1e-10)
                new.append(np.array(pose))
        if new:
            begin=len(U);U=unique_poses(list(U)+new);stage=np.r_[stage,np.full(len(U)-begin,2)];values=sc.score(U)
        # Both material models get identical starts and budgets; their outputs are shared.
        starts=np.unique(values[:,2:].argmin(0));new=[]
        for h in starts:
            for col in [2,3]:
                t,path=joint_fit(sc,U[h],cfg,col);new.append((t,col+1));paths.append((h,col+4,2,path));counters['independent_fits' if col==2 else 'shared_fits']+=1;counters['joint_improved']+=int(path[-1]<path[0]-1e-10)
        for t,kind in new:
            if not any(np.max(abs(t-r))<1e-9 for r in U):U=np.r_[U,t[None]];stage=np.r_[stage,kind]
        values=sc.score(U)
        for m in range(2):
            h=int(values[:,m].argmin());oldm=m+2
            if cc[oldm]<=values[h,m]:cc[m+4]=cc[oldm];TT[m+4]=TT[oldm];origin[m+4]=origin[oldm]
            else:cc[m+4]=values[h,m];TT[m+4]=U[h];origin[m+4]=h
        for col in [2,3]:h=int(values[:,col].argmin());cc[col+4]=values[h,col];TT[col+4]=U[h];origin[col+4]=h
        # Old Cartesian modes are NOT truncated for the new material likelihoods.
        # Nonnegative remaining pair/factor terms give a safe lower bound.
        A,av=pair(0,i,j);B,bv=pair(1,i,k);aidx,bidx=np.indices((len(A),len(B)));lower=(av[:,None,3]+bv[None,:,3]).ravel();aidx=aidx.ravel();bidx=bidx.ravel();order=np.argsort(lower,kind='stable')
        for pos in range(0,len(order),32):
            hh=order[pos:pos+32];hh=hh[(lower[hh,None]<=cc[None,6:]+1e-12).any(1)]
            if not len(hh):continue
            candidate=np.stack([A[aidx[hh]],B[bidx[hh]]],1);vv=sc.score(candidate)[:,2:];actual+=len(hh)
            for m in range(2):
                h=int(vv[:,m].argmin())
                if vv[h,m]<cc[m+6]:cc[m+6]=vv[h,m];TT[m+6]=candidate[h];origin[m+6]=-2-int(hh[h])
        allproposals[f'{q}_transforms']=U;allproposals[f'{q}_stage']=stage;allproposals[f'{q}_values']=values
        for h,(initial,col,e,path) in enumerate(paths):fitrecords[f'{q}_{h}_path']=path;fitrecords[f'{q}_{h}_meta']=np.array([initial,col,e])
        rows.append((cc,TT,origin))
        if profile:return dict(seconds=time.perf_counter()-start,triples=1,candidates=len(U),old_cartesian=len(A)*len(B),actual_material_cartesian=actual,counters=counters)
        if q%8==7:
            save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),elapsed_s=time.perf_counter()-start,counters=counters,actual_material_cartesian=actual))
        if time.perf_counter()-start>cfg['hard_timeout_s']:raise TimeoutError(f'{v}: configured hard timeout')
    costs=np.array([r[0] for r in rows]);lookup={tuple(t):h for h,t in enumerate(old['triples'])}
    values=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8)) for s in graph['states']]);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_proposals.npz',**allproposals);np.savez_compressed(root/f'{v}_fits.npz',**fitrecords)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=costs,transforms=np.array([r[1] for r in rows]),origins=np.array([r[2] for r in rows]))
    np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=values,best=best)
    for col,m in enumerate(METHODS):
        h=int(best[col]);state=graph['states'][h];indices=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(state)]
        np.savez_compressed(root/f'{v}_{m}.npz',target_slots=state,transforms=np.array([rows[q][1][col] for q in indices]),objectives=costs[indices,col],search_state=h,triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,seconds=time.perf_counter()-start,counters=counters,actual_material_cartesian=actual,triples=len(rows),states=len(graph['states']),private_labels_read=False,same_material_candidates=True,equal_flops=False))
    print('V12 DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();cfg=configuration()
    if args.profile:print(json.dumps(job(cfg,'rich',True),indent=2));return
    root=Path(cfg['output']);assert not root.exists(),str(root)
    parent=Path(cfg['retained_root']);integrity=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for g in ['source_hashes','evidence_hashes']:
        for p,h in integrity[g].items():assert sha(p)==h,p
    root.mkdir(parents=True);source_paths=[__file__,'configs/material_inverse_plane_pose_proposals_v12.json','inspection_decisions/material_inverse_plane_pose_proposals_v12.py','scripts/check_material_inverse_plane_pose_proposals_v12.py','scripts/score_material_inverse_plane_pose_proposals_v12.py']
    sources={p:sha(p) for p in source_paths};save(root/'frozen_protocol.json',dict(config=cfg,methods=METHODS,sources=sources,parent_completion_sha=sha(parent/'completion_integrity.json'),scope=cfg['contract']))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as executor:
        for future in as_completed([executor.submit(job,cfg,v) for v in cfg['variants']]):future.result()
    for p,h in sources.items():assert sha(p)==h,p
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,seconds=time.perf_counter()-start,sources=sources,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')}));print('V12 SEALED',flush=True)


if __name__=='__main__':main()
