"""Same-input all-identity comparison with actual new continuous surface proposals."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_surface_patch_registration_v9 import SurfaceScore,icp_fit,projective_fit
from inspection_decisions.material_partial_track_transport_v7 import inverse,PAIRS
from inspection_decisions.material_joint_id_pose_search_v5 import legal_neighbors
from scripts.cache_material_surface_geometry_v9 import unpack
from scripts.run_material_observation_registration_v7 import prepare,make_scorer
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['ordinary_native','surface_geometry','surface_rgb_fusion','projective_points','material_patch']

def setup(cfg,v):
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));rc=json.loads(Path(cfg['registration_config']).read_text(encoding='utf-8'));nc=json.loads(Path(ic['native_config']).read_text(encoding='utf-8'));frames,ids,bc,ep,groups,views=prepare(ic,rc,cfg['condition'],v)
    banks=unpack(Path(cfg['geometry_cache'])/f'{v}_surfaces.npz')
    with np.load(Path(ic['native_root'])/cfg['condition']/f'{v}_edges.npz') as f:edges=unpack_edges(dict(f),nc)
    return ic,frames,ids,bc,ep,groups,views,banks,edges

class AllScore:
    def __init__(self,ordinary,surface):self.ordinary=ordinary;self.surface=surface
    def score(self,T):return np.c_[self.ordinary.score(T),self.surface.score(T)]

def make_all(pe,i,j,bc,ep,groups,views,banks,edges,cfg):
    e,f=PAIRS[pe];return AllScore(make_scorer(pe,i,j,ep,groups,views,edges,bc),SurfaceScore(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,cfg))

def job(cfg,v):
    torch.set_num_threads(1);start=time.perf_counter();root=Path(cfg['output']);parent=Path(cfg['parent_root']);ic,frames,ids,bc,ep,groups,views,banks,edges=setup(cfg,v);pool={};scorers={};pairvalues={};payload={};counts=dict(icp_fits=0,colored_icp_fits=0,colored_no_correspondences=0,projective_fits=0,projective_improved=0)
    def progress(message,**kw):save(root/f'{v}_progress.json',dict(message=message,elapsed_s=time.perf_counter()-start,**kw))
    with np.load(parent/f'{v}_pools.npz') as f:previous=dict(f)
    for pe,(e,f) in enumerate(PAIRS):
        with np.load(Path(cfg['geometry_cache'])/f'pair{pe}_geometry.npz') as packed:global_poses=packed['transforms']
        for i in range(24):
            for j in range(24):
                key=pe,i,j;n=previous['counts'][key];pp=list(previous['transforms'][key][:n]);new=[];scorer=make_all(pe,i,j,bc,ep,groups,views,banks,edges,cfg);scorers[key]=scorer
                for T in global_poses[i,j]:
                    if np.isfinite(T).all() and not any(np.max(abs(T-U))<1e-9 for U in pp):pp.append(T)
                pre=scorer.score(np.array(pp));starts=[]
                for h in [int(pre[:,1].argmin()),int(pre[:,2].argmin())]:
                    if h not in starts:starts.append(h)
                for n,h in enumerate(starts[:cfg['fit_starts']]):
                    T=pp[h]
                    for kind in range(3):
                        if kind==2:U,path=projective_fit(scorer.surface,T,cfg);counts['projective_fits']+=1;counts['projective_improved']+=int(path[-1]<path[0]-1e-10)
                        else:
                            U,failed=icp_fit(banks[e][i],banks[f][j],T,cfg,colored=bool(kind));path=np.zeros(0);counts['colored_icp_fits' if kind else 'icp_fits']+=1;counts['colored_no_correspondences']+=int(failed)
                        assert np.isfinite(U).all();payload[f'{pe}_{i}_{j}_{n}_{kind}_T']=U;payload[f'{pe}_{i}_{j}_{n}_{kind}_initial']=T;payload[f'{pe}_{i}_{j}_{n}_{kind}_path']=path;new.append(U)
                for U in new:
                    if not any(np.max(abs(U-T))<1e-9 for T in pp):pp.append(U)
                pool[key]=np.array(pp);pairvalues[key]=scorer.score(pool[key])
            if i%4==3:progress('surface/ColoredICP/projective proposals',pair=pe,source_slot=i,counters=counts)
        print('SURFACE PROPOSALS',v,pe,round(time.perf_counter()-start,1),flush=True)
    cap=max(map(len,pool.values()));poses=np.tile(np.eye(4),(3,24,24,cap,1,1));sizes=np.zeros((3,24,24),int)
    for key,T in pool.items():poses[key][:len(T)]=T;sizes[key]=len(T)
    np.savez_compressed(root/f'{v}_pools.npz',transforms=poses,counts=sizes,old_counts=previous['counts']);np.savez_compressed(root/f'{v}_fits.npz',**payload)
    with np.load(parent/f'{v}_cache.npz') as f:pcache=dict(f)
    plook={tuple(t):h for h,t in enumerate(pcache['triples'])};cache={};count=0;surface_count=0
    def objective(i,j,k):
        nonlocal count,surface_count
        key=i,j,k
        if key not in cache:
            A,B=pool[0,i,j],pool[1,i,k];T1=np.repeat(A,len(B),0);T2=np.tile(B,(len(A),1,1));relative=T2@inverse(T1);lower=(pairvalues[0,i,j][:,None,:]+pairvalues[1,i,k][None,:,:]).reshape(-1,5);ov=lower[:,0]+.5*scorers[2,j,k].ordinary.score(relative);h=int(ov.argmin());cc=np.r_[ov[h],np.full(4,np.inf)];best=np.full(5,h,int)
            # Every surface term is >=0 (up to floating-point roundoff).
            # Thus H1+H2 is a safe bound, unlike arbitrary pose top-k pruning.
            seeds=np.unique(np.r_[lower[:,1:].argmin(0),h]);vv=lower[seeds,1:]+.5*scorers[2,j,k].surface.score(relative[seeds]);surface_count+=len(seeds)
            for col in range(4):q=int(vv[:,col].argmin());cc[col+1]=vv[q,col];best[col+1]=seeds[q]
            order=np.argsort(lower[:,1:].min(1),kind='stable')
            for start in range(0,len(order),32):
                hh=order[start:start+32];active=(lower[hh,1:]<=cc[None,1:]+1e-12).any(1);hh=hh[active]
                if not len(hh):continue
                vv=lower[hh,1:]+.5*scorers[2,j,k].surface.score(relative[hh]);surface_count+=len(hh)
                for col in range(4):
                    q=int(vv[:,col].argmin());candidate=float(vv[q,col]);m=col+1
                    if candidate<cc[m] or (candidate==cc[m] and hh[q]<best[m]):cc[m]=candidate;best[m]=hh[q]
            TT=np.array([[T1[h],T2[h]] for h in best]);origins=np.zeros(5,int)
            # Keep old coupled endpoints paired, not synthetic cross-mode combinations.
            if key in plook:
                oldT=pcache['transforms'][plook[key]];vold=scorers[0,i,j].score(oldT[:,0])+scorers[1,i,k].score(oldT[:,1])+.5*scorers[2,j,k].score(oldT[:,1]@inverse(oldT[:,0]));hh=vold.argmin(0)
                for col,h in enumerate(hh):
                    if vold[h,col]<cc[col]:cc[col]=vold[h,col];TT[col]=oldT[h];origins[col]=h+1
            cache[key]=(cc,TT,origins);count+=len(T1)
            if len(cache)%8==0:progress('exact bounded Cartesian joint identity/pose',triples=len(cache),pose_pairs=count,surface_pairs=surface_count)
        return cache[key]
    with np.load(parent/f'{v}_search_graph.npz') as f:initials=list(f['states'])
    costs=[]
    for col in range(5):
        c={(e,f):np.array([[pairvalues[pe,i,j][:,col].min() for j in range(24)] for i in range(24)]) for pe,(e,f) in enumerate(PAIRS)};costs.append(c);a,b=old.solve_assignments(c,bc);initials.append(np.c_[a,b])
    states=[];values=[];seen=set()
    def add(s):
        s=np.asarray(s,np.int64)
        if s.tobytes() in seen:return False
        seen.add(s.tobytes());states.append(s.copy());values.append(sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(s)),start=np.zeros(5)));return True
    for s in initials:add(s)
    initial_count=len(states);old_allowed=[[{int(s[i,e]) for s in states} for i in range(24)] for e in range(2)];proxy=[costs[3][p] for p in PAIRS];allowed=[[old_allowed[e][i]|set(np.argsort(proxy[e][i])[:3].tolist()) for i in range(24)] for e in range(2)];proposals=[]
    for h in np.unique(np.array(values).argmin(0)):proposals+=legal_neighbors(states[h],allowed,proxy,old_allowed)
    proposals.sort(key=lambda x:(x[0],x[1],x[2].tobytes()));added=0
    for _,__,s in proposals:
        added+=int(add(s))
        if added>=cfg['extra_identity_states']:break
    values=np.array(values);states=np.array(states);best=values.argmin(0);np.savez_compressed(root/f'{v}_search_graph.npz',states=states,values=values,best=best,initial_count=initial_count);np.savez_compressed(root/f'{v}_cache.npz',triples=np.array(list(cache)),costs=np.array([r[0] for r in cache.values()]),transforms=np.array([r[1] for r in cache.values()]),origins=np.array([r[2] for r in cache.values()]))
    for col,m in enumerate(METHODS):
        h=best[col];s=states[h];TT=np.array([objective(i,int(j),int(k))[1][col] for i,(j,k) in enumerate(s)]);cc=np.array([objective(i,int(j),int(k))[0][col] for i,(j,k) in enumerate(s)]);np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=TT,objectives=cc,search_state=h,triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,elapsed_s=time.perf_counter()-start,counters=counts,states=len(states),triples=len(cache),cartesian_pose_pairs=count,actual_surface_pair_scores=surface_count,private_labels_read=False,same_inputs_and_final_candidates=True,equal_flops=False));print('SURFACE REGISTRATION DONE',v,round(time.perf_counter()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');ap.add_argument('--output');args=ap.parse_args();cp=Path('configs/material_surface_patch_registration_v9.json');cfg=json.loads(cp.read_text(encoding='utf-8'))
    if args.output:cfg['output']=args.output
    if args.profile:
        _,_,_,bc,ep,groups,views,banks,edges=setup(cfg,'rich');s=make_all(0,0,0,bc,ep,groups,views,banks,edges,cfg)
        with np.load(Path(cfg['parent_root'])/'rich_pools.npz') as f:T=f['transforms'][0,0,0,:f['counts'][0,0,0]]
        poses=np.tile(T,(16,1,1));start=time.perf_counter();s.score(poses);elapsed=time.perf_counter()-start;start=time.perf_counter();icp_fit(banks[0][0],banks[1][0],T[0],cfg,True);U,path=projective_fit(s.surface,T[0],cfg);print(json.dumps(dict(public_only=True,poses=len(poses),score_seconds=elapsed,fit_seconds=time.perf_counter()-start,fit_path=path.tolist())));return
    geometry=Path(cfg['geometry_cache']);seal=json.loads((geometry/'surface_seal.json').read_text(encoding='utf-8'))
    for p,h in seal['sources'].items():assert sha(p)==h
    for p,h in seal['files'].items():assert sha(geometry/p)==h
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);sources={p:sha(p) for p in [__file__,str(cp),'inspection_decisions/material_surface_patch_registration_v9.py','scripts/check_material_surface_patch_registration_v9.py','scripts/cache_material_surface_geometry_v9.py',cfg['input_config'],cfg['registration_config']]};start=time.perf_counter();save(root/'frozen_protocol.json',dict(config=cfg,methods=METHODS,sources=sources,surface_seal_sha=sha(geometry/'surface_seal.json'),private_labels_read=False,scope='One TRAIN, all24x4, no network training. Common FGR/ICP/ColoredICP/projective candidates, finite common identity graph, exact Cartesian scoring on visited triples. Old coupled endpoints retained. Open3D ColoredICP is a proposal backend, NOT a standalone full published system benchmark.'))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as executor:
        for f in as_completed([executor.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')}));print('SURFACE REGISTRATION SEALED',flush=True)

if __name__=='__main__':main()
