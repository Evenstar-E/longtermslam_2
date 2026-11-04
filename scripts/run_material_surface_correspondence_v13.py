"""Full24 explicit chart correspondences; common finite coupled candidate graph."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.prepare_material_surface_charts_v13 import configuration,setup
from scripts.cache_material_surface_geometry_v9 import unpack
from inspection_decisions.material_surface_correspondence_v13 import ChartPair,ChartTriple,fit_correspondences,PAIRS
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import unique_poses,propagation
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['retained_v12','multiscale_fusion','partial_points','shared_points','partial_normals','shared_normals','partial_phase','shared_phase']
ACTUAL_SCOPE=('All24x4 on seenTRAIN. Common measured64-node charts and common finite coupled candidate union for all7new scorers. '
 'Retain EVERY V12 generated coupled candidate and every actual V12 per-triple selected endpoint, including old Cartesian winners; '
 'V12 exact predictions replayed separately. New objectives do NOT exhaustively re-search all unselected V9 Cartesian combinations. '
 'This explicitly supersedes the planned exhaustive-Cartesian statement in the chart-cache config, which only prepared observations. '
 'Fixed same V12 finite identity graph. More queried texture pixels thanV12, not same-FLOP or pure information-budget comparison. '
 'Same-feature points/normal/phase independent/shared controls isolate actual mechanism. No reference poses/IDs/material labels before prediction seal.')


def load(path):
    with np.load(path) as f:return dict(f)


class AllScore:
    def __init__(self,pairs,cfg):self.pairs=pairs;self.core=ChartTriple(pairs,cfg)
    def score(self,T):
        values=[]
        for t in T:
            c=self.core.evaluate(t);u=[t[0],t[1],t[1]@inverse(t[0])];f=sum(w*p.fusion(x) for p,x,w in zip(self.pairs,u,[1,1,.5]))/2.5;values.append(np.r_[f,c])
        return np.array(values)


def job(cfg,v,profile=False):
    torch.set_num_threads(1);root=Path(cfg['output']);parent=Path(cfg['previous_root']);start=time.perf_counter();_,frames,_,_,_,_,views,_,_=setup(cfg,v);charts=unpack(Path(cfg['chart_cache'])/f'{v}_charts.npz')
    old=load(parent/f'{v}_cache.npz');graph=load(parent/f'{v}_search_graph.npz');generated=load(parent/f'{v}_proposals.npz');pools=load(Path(cfg['parent_root'])/f'{v}_pools.npz')
    @lru_cache(maxsize=128)
    def pair(pe,i,j):
        e,f=PAIRS[pe];return ChartPair(charts[e][i],charts[f][j],views[e][i],views[f][j],i,j,cfg)
    proposals={};pairpayload={};diagnostics=[];fits={};rows=[];counters=dict(pair_families=0,ordinary_pose_trials=0,frame_pose_trials=0,continuous_fits=0,continuous_improved=0,new_pair_endpoints=0)
    @lru_cache(maxsize=512)
    def global_modes(pe,i,j):
        p=pair(pe,i,j);a,b,stat=p.proposals(cfg['seed']+pe*576+i*24+j);T=unique_poses(list(a)+list(b));counters['pair_families']+=1;counters['ordinary_pose_trials']+=stat['ordinary_trials'];counters['frame_pose_trials']+=stat['frame_trials'];counters['new_pair_endpoints']+=len(T)
        prefix=f'{pe}_{i}_{j}';pairpayload[prefix+'_poses']=T;pairpayload[prefix+'_ordinary']=a;pairpayload[prefix+'_frame']=b;pairpayload[prefix+'_counts']=np.array([stat[k] for k in ['ordinary_trials','frame_trials','all_correspondence_edges','proposal_edges']])
        # A real public-input ranking over all old pair modes + both new families.
        oldT=pools['transforms'][pe,i,j,:pools['counts'][pe,i,j]];allT=unique_poses(list(oldT)+list(T));pv=[]
        for t in allT:
            ev=[p.evidence(t,o) for o in [0,1,2]];pv.append([p.fusion(t),*[(r['potential']-r['value'])/r['potential']+cfg['free_weight']*r['free']/2.5 for r in ev]])
        h=np.unique(np.array(pv).argmin(0));return T,allT[h]
    triples=old['triples'][:1] if profile else old['triples']
    for q,(i,j,k) in enumerate(triples):
        i,j,k=map(int,(i,j,k));ps=[pair(0,i,j),pair(1,i,k),pair(2,j,k)];sc=AllScore(ps,cfg)
        ga,sa=global_modes(0,i,j);gb,sb=global_modes(1,i,k);gc,ss=global_modes(2,j,k)
        retained=unique_poses(list(generated[f'{q}_transforms'])+list(old['transforms'][q]))
        # Pair phase/RANSAC outputs are kept even when old local scores prefer
        # another mode; all six new H1xH2 combinations enter the common graph.
        direct=np.array([[a,b] for a in ga for b in gb]);composed=propagation(sa,sb,ss)
        U=unique_poses(list(retained)+list(direct)+list(composed));stage=np.array([0 if h<len(retained) else 1 for h in range(len(U))]);values=sc.score(U)
        seeds=np.unique(values[:,[0,5]].argmin(0));new=[];fitmeta=[]
        for h in seeds:
            for col in [0,1,4,5]:
                T,path=fit_correspondences(sc.core,U[h],cfg,col);new.append(T);fitmeta.append((int(h),col,path));counters['continuous_fits']+=1;counters['continuous_improved']+=int(path[-1]<path[0]-1e-10)
        previous=len(U);U=unique_poses(list(U)+new);stage=np.r_[stage,np.full(len(U)-previous,2)];values=sc.score(U);h=values.argmin(0);cc=np.r_[old['costs'][q,6],values[h,np.arange(7)]];TT=np.r_[old['transforms'][q,6][None],U[h]]
        proposals[f'{q}_transforms']=U;proposals[f'{q}_stage']=stage;proposals[f'{q}_values']=values
        for h0,(initial,col,path) in enumerate(fitmeta):fits[f'{q}_{h0}_path']=path;fits[f'{q}_{h0}_meta']=np.array([initial,col])
        rows.append((cc,TT));diagnostics.append(dict(triple=[i,j,k],retained=len(retained),global_and_composed=len(U)-len(retained),final_candidates=len(U)))
        if profile:return dict(seconds=time.perf_counter()-start,candidates=len(U),retained=len(retained),counters=counters,private_labels_read=False)
        if q%8==7:save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,counters=counters))
        if time.perf_counter()-start>cfg['hard_timeout_s']:raise TimeoutError(v)
    costs=np.array([r[0] for r in rows]);lookup={tuple(t):q for q,t in enumerate(old['triples'])};values=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8)) for s in graph['states']]);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=costs,transforms=np.array([r[1] for r in rows]));np.savez_compressed(root/f'{v}_proposals.npz',**proposals);np.savez_compressed(root/f'{v}_pair_proposals.npz',**pairpayload);np.savez_compressed(root/f'{v}_fits.npz',**fits);np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=values,best=best)
    for col,m in enumerate(METHODS):
        state=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(state)]
        np.savez_compressed(root/f'{v}_{m}.npz',target_slots=state,transforms=np.array([rows[q][1][col] for q in ix]),objectives=costs[ix,col],search_state=best[col],triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,seconds=time.perf_counter()-start,counters=counters,triples=len(rows),states=len(graph['states']),candidate_counts=diagnostics,private_labels_read=False,scope=ACTUAL_SCOPE));print('V13 DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();cfg=configuration();cfg['actual_scope']=ACTUAL_SCOPE;cfg['contract']=ACTUAL_SCOPE
    if args.profile:print(json.dumps(job(cfg,'rich',True),indent=2));return
    root=Path(cfg['output']);assert not root.exists();charts=Path(cfg['chart_cache']);seal=json.loads((charts/'chart_seal.json').read_text(encoding='utf-8'))
    for p,h in seal['sources'].items():assert sha(p)==h,p
    for p,h in seal['files'].items():assert sha(charts/p)==h,p
    parent=Path(cfg['previous_root']);assert sha(parent/'completion_integrity.json')==seal['parent_completion_sha']
    root.mkdir(parents=True);paths=[__file__,'scripts/score_material_surface_correspondence_v13.py','scripts/check_material_surface_correspondence_v13.py','inspection_decisions/material_surface_correspondence_v13.py','configs/material_surface_correspondence_v13.json'];sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=cfg,sources=sources,methods=METHODS,scope=ACTUAL_SCOPE,chart_seal_sha=sha(charts/'chart_seal.json'),parent_completion_sha=sha(parent/'completion_integrity.json')));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')},seconds=time.perf_counter()-start));print('V13 SEALED',flush=True)


if __name__=='__main__':main()
