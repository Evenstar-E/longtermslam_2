"""All512 measured centers, same-view dense controls and continuous material UV."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse,py_compile
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration as old_configuration,setup
from inspection_decisions.material_continuous_surface_correspondence_v14 import DensePair,ContinuousTriple,fit_pose,spectral_proposals,PAIRS
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import unique_poses,propagation
from inspection_decisions.textured_material_matching_v1 import world_points,apply,rigid
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['retained_v12','dense_geometry','dense_fusion','projection_fixed','continuous_pair','material_independent','material_track_independent','material_track_shared']


def configuration():
    cfg=old_configuration();cfg.update(json.loads(Path('configs/material_continuous_surface_correspondence_v14.json').read_text(encoding='utf-8')));return cfg


def load(path):
    with np.load(path) as f:return dict(f)


def dense_records(views):
    out=[]
    for row in views:
        result=[]
        for slot,frames in enumerate(row):
            xyz=[];rgb=[];uv=[];fi=[]
            for h,f in enumerate(frames):
                y,x=np.nonzero(f['depth_valid']&(f['anonymous_slot']==slot));p=np.c_[x,y];xyz.append(world_points(f,p));rgb.append(f['rgb'][y,x]/255.);uv.append(p);fi.append(np.full(len(p),h))
            result.append(dict(xyz=np.concatenate(xyz),rgb=np.concatenate(rgb),uv=np.concatenate(uv),frame=np.concatenate(fi)))
        out.append(result)
    return out


def dense_icp(pair,T,cfg,fusion):
    T=T.copy()
    for _ in range(cfg['pose_steps']):
        x=apply(T,pair.a['xyz']);dist,idx=pair.tb.query(x);ok=dist<cfg['continuous_capture_m']
        if ok.sum()<3:break
        weight=np.exp(-.5*(dist[ok]/cfg['continuous_sigma_m'])**2)
        if fusion:weight*=np.exp(-.5*np.mean((pair.a['rgb'][ok]-pair.db['rgb'][idx[ok]])**2,axis=1)/(cfg['source_color_sigma']**2+cfg['target_color_sigma']**2))
        U=rigid(pair.a['xyz'][ok],pair.db['xyz'][idx[ok]],weight)
        if U is None:break
        if np.max(abs(U-T))<1e-9:break
        T=U
    return T


def job(cfg,v,profile=False):
    torch.set_num_threads(1);root=Path(cfg['output']);start=time.perf_counter();_,frames,_,_,_,_,views,banks,_=setup(cfg,v);dense=dense_records(views)
    parent=Path(cfg['retained_root']);old=load(parent/f'{v}_cache.npz');graph=load(parent/f'{v}_search_graph.npz');gen=load(parent/f'{v}_proposals.npz');v13=load(Path(cfg['previous_root'])/f'{v}_cache.npz');pools=load(Path(cfg['parent_root'])/f'{v}_pools.npz');look13={tuple(t):q for q,t in enumerate(v13['triples'])}
    @lru_cache(maxsize=128)
    def pair(pe,i,j):
        e,f=PAIRS[pe];return DensePair(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],cfg)
    counters=dict(spectral_families=0,spectral_models=0,dense_icp_fits=0,continuous_fits=0,continuous_improved=0);pairsaved={};payload={};fits={};rows=[]
    @lru_cache(maxsize=512)
    def modes(pe,i,j):
        p=pair(pe,i,j);raw,stat=spectral_proposals(p.a,p.b,cfg);counters['spectral_families']+=1;counters['spectral_models']+=len(raw)
        oldT=pools['transforms'][pe,i,j,:pools['counts'][pe,i,j]];initial=unique_poses(list(oldT)+list(raw));scores=np.array([p.nearest(t)[:2] for t in initial]);starts=np.unique(scores.argmin(0));new=list(raw)
        for h in starts:
            for fusion in [False,True]:new.append(dense_icp(p,initial[h],cfg,fusion));counters['dense_icp_fits']+=1
        new=unique_poses(new);value=np.array([p.nearest(t)[:2] for t in new]);order=[]
        for col in [0,1]:order+=list(np.argsort(value[:,col],kind='stable')[:cfg['global_keep']])
        selected=new[np.unique(order)];pairsaved[f'{pe}_{i}_{j}_poses']=selected;pairsaved[f'{pe}_{i}_{j}_statistics']=np.array([stat['edges'],stat['seeds'],stat['models']]);return selected
    triples=old['triples'][:1] if profile else old['triples']
    for q,(i,j,k) in enumerate(triples):
        i,j,k=map(int,(i,j,k));ps=[pair(0,i,j),pair(1,i,k),pair(2,j,k)];sc=ContinuousTriple(ps,cfg)
        if profile:
            T=old['transforms'][q,6];t=time.perf_counter();value=sc.score(T[None]);seconds=time.perf_counter()-t
            return dict(seconds=time.perf_counter()-start,one_triple_score_s=seconds,values=value.tolist(),all_source_counts=[len(p.a['xyz']) for p in ps],dense_pixels=sum(len(a['xyz']) for row in dense for a in row),private_labels_read=False)
        A,B,C=modes(0,i,j),modes(1,i,k),modes(2,j,k);retained=list(gen[f'{q}_transforms'])+list(old['transforms'][q]);key=i,j,k
        if key in look13:retained+=list(v13['transforms'][look13[key]])
        retained=unique_poses(retained);new=list(np.array([[a,b] for a in A for b in B]))+list(propagation(A[:2],B[:2],C[:2]));U=unique_poses(list(retained)+new);stage=np.r_[np.zeros(len(retained),int),np.ones(len(U)-len(retained),int)];value=sc.score(U)
        # Shared same ordinary-continuous starting point; both outputs are
        # added to the common final graph, no method owns a good endpoint.
        initial=int(value[:,3].argmin());new=[]
        for col in [3,6]:
            T,path=fit_pose(sc,U[initial],cfg,col);new.append(T);fits[f'{q}_{col}_initial']=U[initial];fits[f'{q}_{col}_T']=T;fits[f'{q}_{col}_path']=path;counters['continuous_fits']+=1;counters['continuous_improved']+=int(path[-1]<path[0]-1e-10)
        n=len(U);U=unique_poses(list(U)+new);stage=np.r_[stage,np.full(len(U)-n,2)];value=sc.score(U);h=value.argmin(0);cost=np.r_[old['costs'][q,6],value[h,np.arange(7)]];T=np.r_[old['transforms'][q,6][None],U[h]];rows.append((cost,T));payload[f'{q}_transforms']=U;payload[f'{q}_values']=value;payload[f'{q}_stage']=stage
        if q%8==7:
            # Unique completed blocks survive an interrupted long CPU run.
            np.savez_compressed(root/f'{v}_block_{q-7:04}.npz',triples=old['triples'][q-7:q+1],costs=np.array([r[0] for r in rows[-8:]]),transforms=np.array([r[1] for r in rows[-8:]]),**{key:val for key,val in payload.items() if q-7<=int(key.split('_')[0])<=q})
            save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,counters=counters))
        if time.perf_counter()-start>cfg['hard_timeout_s']:raise TimeoutError(v)
    cost=np.array([r[0] for r in rows]);lookup={tuple(t):q for q,t in enumerate(old['triples'])};values=np.array([sum((cost[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8)) for s in graph['states']]);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=cost,transforms=np.array([r[1] for r in rows]));np.savez_compressed(root/f'{v}_proposals.npz',**payload);np.savez_compressed(root/f'{v}_pair_proposals.npz',**pairsaved);np.savez_compressed(root/f'{v}_fits.npz',**fits);np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=values,best=best)
    for col,m in enumerate(METHODS):
        s=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(s)];np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=np.array([rows[q][1][col] for q in ix]),objectives=cost[ix,col],search_state=best[col],triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,seconds=time.perf_counter()-start,counters=counters,triples=len(rows),states=len(graph['states']),source_centers=sum(len(a['xyz']) for row in banks for a in row),dense_target_pixels=sum(len(a['xyz']) for row in dense for a in row),private_labels_read=False,scope=cfg['contract']));print('V14 DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();cfg=configuration()
    if args.profile:print(json.dumps(job(cfg,'rich',True),indent=2));return
    parent=Path(cfg['previous_root']);index=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in index[group].items():assert sha(p)==h,p
    paths=[__file__,'inspection_decisions/material_continuous_surface_correspondence_v14.py','configs/material_continuous_surface_correspondence_v14.json','scripts/check_material_continuous_surface_correspondence_v14.py','scripts/score_material_continuous_surface_correspondence_v14.py']
    for p in paths:
        if str(p).endswith('.py'):py_compile.compile(p,doraise=True)
    root=Path(cfg['output']);assert not root.exists();root.mkdir(parents=True);sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=cfg,methods=METHODS,sources=sources,parent_completion_sha=sha(parent/'completion_integrity.json'),scope=cfg['contract']));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')},seconds=time.perf_counter()-start));print('V14 SEALED',flush=True)


if __name__=='__main__':main()
