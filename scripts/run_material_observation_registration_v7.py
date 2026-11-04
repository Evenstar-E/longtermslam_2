"""Common-input ordinary baseline and partial pair registration on fresh TRAIN."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_observation_inputs_v7 import load_epochs
from inspection_decisions.material_native_matching_frontend_v6 import selected_views
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges,pair_modes,material_tracks
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_joint_id_pose_search_v5 import legal_neighbors
from inspection_decisions.material_partial_pair_registration_v7 import fit
from scripts.run_material_native_modes_v6 import NativeScore,merge_poses
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['base_pool','native_ransac','native_graph','partial_joint'];PAIRS=[(0,1),(0,2),(1,2)]


def prepare(cfg,rcfg,c,v):
    frames,ids=load_epochs(cfg,c,v);bcfg=json.loads(Path(cfg['legacy_config']).read_text(encoding='utf-8'));ep=[old.extract_epoch(f,bcfg,cfg['seed']+e) for e,f in enumerate(frames)];groups=[];views=[]
    for e in range(3):
        gg=[];vv=[]
        for slot,a in enumerate(ep[e]):
            assert len(a['points'])>0,(c,v,e,slot,'No measured support; need explicit unknown baseline rather than NaN')
            ix=np.linspace(0,len(a['points'])-1,min(rcfg['ray_queries'],len(a['points'])),dtype=int);gg.append(dict(xyz=a['points'][ix],area=np.ones(len(ix))/len(ix)))
            vv.append([frames[e][k] for k in selected_views(frames[e],slot,rcfg['score_views_per_slot'])])
        groups.append(gg);views.append(vv)
    return frames,ids,bcfg,ep,groups,views


def make_scorer(pe,i,j,ep,groups,views,edges,bcfg):
    e,f=PAIRS[pe];a,b=ep[e][i],ep[f][j]
    return NativeScore(PairScore(a,b,old.matches(a,b,bcfg),groups[e][i],groups[f][j],{},views[f][j],j,bcfg),edges[pe][i*24+j])


def common_search(initials,objective,allowed,cost,old_allowed,cfg):
    states=[];values=[];seen=set();paths=[]
    def add(s):
        s=np.asarray(s,dtype=np.int64)
        if s.tobytes() in seen:return
        seen.add(s.tobytes());states.append(s.copy());values.append(sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(s)),start=np.zeros(4)))
    for s in initials:add(s)
    initial=len(states)
    for _ in range(cfg['identity_rounds']):
        proposals=[]
        for q in np.unique(np.array(values).argmin(0)):proposals+=legal_neighbors(states[q],allowed,cost,old_allowed)
        proposals.sort(key=lambda x:(x[0],x[1],x[2].tobytes()));n=0
        for _,__,s in proposals:
            if s.tobytes() in seen:continue
            add(s);n+=1
            if n>=cfg['identity_neighbors']:break
        paths.append(np.array(values).min(0))
        if not n:break
    values=np.array(values);return dict(states=np.array(states),values=values,best=values.argmin(0),paths=np.array(paths),initial=initial)


def job(cfg,rcfg,ncfg,c,v):
    torch.set_num_threads(1);root=Path(cfg['output'])/c;start=time.perf_counter()
    def progress(msg,**kw):save(root/f'{v}_progress.json',dict(message=msg,elapsed_s=time.perf_counter()-start,**kw))
    frames,ids,bcfg,ep,groups,views=prepare(cfg,rcfg,c,v);base=old.geometric_pools(ep,bcfg)
    with np.load(Path(cfg['native_root'])/c/f'{v}_edges.npz') as f:edges=unpack_edges(dict(f),ncfg)
    pools={};masks={};scorers={};edgevalues={}
    for pe,(e,f) in enumerate(PAIRS):
        for i in range(24):
            for j in range(24):
                a,b=ep[e][i],ep[f][j];match=old.matches(a,b,bcfg);T=old.ransac(a,b,match,bcfg,cfg['seed']+pe*10000+i*24+j);parent=list(base[e,f][i][j])
                if T is not None:parent += [T,old.icp(a,b,T,bcfg)]
                r=edges[pe][i*24+j];ras=pair_modes(r,ncfg,'ransac',cfg['seed']+pe*10000+i*24+j);graph=pair_modes(r,ncfg,'graph',0);ras=ras+[old.icp(a,b,T,bcfg) for T in ras];graph=graph+[old.icp(a,b,T,bcfg) for T in graph]
                pp,mm=merge_poses(parent,ras,graph);key=pe,i,j;pools[key]=pp;masks[key]=mm;scorers[key]=make_scorer(pe,i,j,ep,groups,views,edges,bcfg);edgevalues[key]=scorers[key].score(pp)
        progress('pair pools '+str((e,f)));print('V7 PAIR POOLS',c,v,e,f,flush=True)
    cap=max(map(len,pools.values()));transforms=np.tile(np.eye(4),(3,24,24,cap,1,1));counts=np.zeros((3,24,24),int);family=np.zeros((3,3,24,24,cap),bool)
    for (pe,i,j),pp in pools.items():transforms[pe,i,j,:len(pp)]=pp;counts[pe,i,j]=len(pp);family[:,pe,i,j,:len(pp)]=masks[pe,i,j]
    np.savez_compressed(root/f'{v}_pools.npz',transforms=transforms,counts=counts,family_masks=family)
    cache={};payload={};fitcount=0
    def objective(i,j,k):
        nonlocal fitcount
        key=i,j,k
        if key not in cache:
            A,B=pools[0,i,j],pools[1,i,k];T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1));value=edgevalues[0,i,j][:,None]+edgevalues[1,i,k][None]+.5*scorers[2,j,k].score(T2@inverse(T1)).reshape(len(A),len(B));cc=[];TT=[]
            for fam in range(3):
                ok=masks[0,i,j][fam,:,None]&masks[1,i,k][fam,None];vv=np.where(ok,value,np.inf);h=int(vv.argmin());cc.append(float(vv.ravel()[h]));TT.append(np.array([T1[h],T2[h]]))
            rr=[edges[0][i*24+j],edges[1][i*24+k],edges[2][j*24+k]];origin=-1;best=cc[2];bestT=TT[2];q=len(cache);n=0
            if sum(len(r['weight'])>=3 for r in rr)>=2:
                allowed=masks[0,i,j][2,:,None]&masks[1,i,k][2,None];order=np.argsort(np.where(allowed,value,np.inf).ravel());seen=[]
                for h in order:
                    if not np.isfinite(value.ravel()[h]) or not allowed.ravel()[h]:break
                    initial=np.array([T1[h],T2[h]])
                    if any(np.max(abs(initial-U))<1e-8 for U in seen):continue
                    seen.append(initial);result,path=fit(initial,rr,cfg['joint_steps']);score=float(scorers[0,i,j].score(result[:1])[0]+scorers[1,i,k].score(result[1:])[0]+.5*scorers[2,j,k].score((result[1]@inverse(result[0]))[None])[0])
                    for field,a in [('initial',initial),('T',result),('path',path)]:payload[f'{q}_{n}_{field}']=a
                    if score<best:best=score;bestT=result;origin=n
                    n+=1;fitcount+=1
                    if n>=cfg['modes']:break
            payload[f'{q}_count']=n;cc.append(best);TT.append(bestT);cache[key]=(np.array(cc),np.array(TT),origin)
            if len(cache)%100==0:progress('joint hypotheses',triples=len(cache),fits=fitcount)
        return cache[key]
    costs={};initials=[]
    for fam in range(3):
        costs[fam]={(e,f):np.array([[edgevalues[pe,i,j][masks[pe,i,j][fam]].min() for j in range(24)] for i in range(24)]) for pe,(e,f) in enumerate(PAIRS)}
        a,b=old.solve_assignments(costs[fam],bcfg);initials.append(np.c_[a,b])
    old_allowed=[[{int(s[i,e]) for s in initials} for i in range(24)] for e in range(2)];proxy=[costs[2][p] for p in PAIRS];allowed=[[old_allowed[e][i]|set(np.argsort(proxy[e][i])[:3].tolist()) for i in range(24)] for e in range(2)]
    tracks,_=material_tracks(edges,ncfg)
    for i,j,k in tracks:allowed[0][i].add(j);allowed[1][i].add(k)
    graph=common_search(initials,objective,allowed,proxy,old_allowed,cfg);np.savez_compressed(root/f'{v}_search_graph.npz',**graph)
    triples=np.array(list(cache));np.savez_compressed(root/f'{v}_cache.npz',triples=triples,costs=np.array([r[0] for r in cache.values()]),transforms=np.array([r[1] for r in cache.values()]),origins=np.array([r[2] for r in cache.values()]));np.savez_compressed(root/f'{v}_partial_fits.npz',triples=triples,**payload)
    for col,m in enumerate(METHODS):
        q=int(graph['best'][col]);slots=graph['states'][q];TT=np.array([objective(i,int(j),int(k))[1][col] for i,(j,k) in enumerate(slots)]);cc=np.array([objective(i,int(j),int(k))[0][col] for i,(j,k) in enumerate(slots)])
        np.savez_compressed(root/f'{v}_{m}.npz',target_slots=slots,transforms=TT,objectives=cc,search_state=q,triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,elapsed_s=time.perf_counter()-start,visited_triples=len(cache),states=len(graph['states']),partial_pair_fits=fitcount,actual_three_track_tuples=len(tracks),raw_frames=[len(f) for f in frames],ray_query_counts=[[len(g['xyz']) for g in gg] for gg in groups],same_inputs_and_score=True,private_labels_read=False))
    print('V7 REGISTRATION',c,v,round(time.perf_counter()-start,1),flush=True)


def main():
    rp=Path('configs/material_observation_registration_v7.json');rcfg=json.loads(rp.read_text(encoding='utf-8'));cfg=json.loads(Path(rcfg['input_config']).read_text(encoding='utf-8'));ncfg=json.loads(Path(cfg['native_config']).read_text(encoding='utf-8'));native=Path(cfg['native_root']);start=time.monotonic()
    while not (native/'correspondence_seal.json').exists():
        if time.monotonic()-start>5400:raise TimeoutError('Native seal absent; no job killed')
        time.sleep(2)
    s=json.loads((native/'correspondence_seal.json').read_text(encoding='utf-8'));assert s['complete']
    for p,h in s['sources'].items():assert sha(p)==h
    for p,h in s['files'].items():assert sha(native/p)==h
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    for c in cfg['conditions']:(root/c).mkdir()
    paths=[__file__,str(rp),rcfg['input_config'],cfg['legacy_config'],'inspection_decisions/material_partial_pair_registration_v7.py','inspection_decisions/material_observation_inputs_v7.py','inspection_decisions/material_correspondence_mode_graph_v6.py','scripts/run_material_native_modes_v6.py'];sources={p:sha(p) for p in paths}
    save(root/'frozen_protocol.json',dict(config=cfg,registration_config=rcfg,native_config=ncfg,methods=METHODS,sources=sources,native_seal_sha=sha(native/'correspondence_seal.json'),dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),score='V6 ordinary geometry/SIFT/ray+native support, ray queries32 uniform from same measured geometry cloud, target top2 observed views; same for4methods',meaning='TRAIN engineering probe; partial_joint is ordinary matched-pair registration, not latent-landmark BA or new persistent-material algorithm'))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,rcfg,ncfg,c,v) for c in cfg['conditions'] for v in cfg['variants']]):f.result()
    assert all(sha(p)==h for p,h in sources.items());save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}));print('V7 REGISTRATION SEALED',flush=True)


if __name__=='__main__':main()
