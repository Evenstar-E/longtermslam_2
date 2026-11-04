"""Frozen full24 public-input comparison; no private labels or target poses."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from inspection_decisions.material_partial_track_transport_v7 import banks_from_features,PairTransport,joint_values,fit,geometry_starts,PAIRS,inverse
from inspection_decisions.material_partial_pair_registration_v7 import fit as ordinary_fit
from inspection_decisions.material_joint_id_pose_search_v5 import legal_neighbors
from scripts.run_material_observation_registration_v7 import prepare,make_scorer
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['ordinary_union','visibility_pair','history_union','shared_transport']

def setup(cfg,v):
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));rc=json.loads(Path(cfg['registration_config']).read_text(encoding='utf-8'));nc=json.loads(Path(ic['native_config']).read_text(encoding='utf-8'))
    frames,ids,bc,ep,groups,views=prepare(ic,rc,cfg['condition'],v)
    with np.load(Path(ic['native_root'])/cfg['condition']/f'{v}_features.npz') as f:banks=banks_from_features(dict(f))
    with np.load(Path(ic['native_root'])/cfg['condition']/f'{v}_edges.npz') as f:edges=unpack_edges(dict(f),nc)
    return ic,bc,ep,groups,views,banks,edges

def job(cfg,v):
    torch.set_num_threads(1);start=time.perf_counter();root=Path(cfg['output']);parent=Path(cfg['parent_root']);ic,bc,ep,groups,views,banks,edges=setup(cfg,v)
    def progress(message,**kw):save(root/f'{v}_progress.json',dict(message=message,elapsed_s=time.perf_counter()-start,**kw))
    with np.load(parent/f'{v}_pools.npz') as f:oldpool=dict(f)
    pool={};scorers={};transport={};ordinary={};paircost={};counts=dict(pair_evidence=0,cartesian_pairs=0,partition_evaluations=0,partition_bound_skips=0,fits=0)
    for pe,(e,f) in enumerate(PAIRS):
        for i in range(24):
            for j in range(24):
                key=pe,i,j;a,b=banks[e][i],banks[f][j];n=int(oldpool['counts'][pe,i,j]);pp=list(oldpool['transforms'][pe,i,j,:n])
                for T in geometry_starts(ep[e][i]['points'],ep[f][j]['points']):
                    if not any(np.max(abs(T-U))<1e-10 for U in pp):pp.append(T)
                pool[key]=np.array(pp);scorers[key]=make_scorer(pe,i,j,ep,groups,views,edges,bc);transport[key]=PairTransport(a,b,edges[pe][i*24+j],views[e][i],views[f][j],i,j,cfg);ordinary[key]=scorers[key].score(pool[key]);vv=[]
                for T in pp:
                    ev=transport[key].evidence(T);counts['pair_evidence']+=1;vv.append(1-ev['value']/(ev['potential']+cfg['support_regularizer'])+cfg['free_weight']*ev['free'])
                paircost[key]=np.array(vv)
            if i%6==5:progress('shared pair pools',pair=pe,source_slot=i)
        print('TRANSPORT PAIRS',v,pe,round(time.perf_counter()-start,1),flush=True)
    cap=max(map(len,pool.values()));pp=np.tile(np.eye(4),(3,24,24,cap,1,1));nn=np.zeros((3,24,24),int)
    for key,values in pool.items():pp[key][:len(values)]=values;nn[key]=len(values)
    np.savez_compressed(root/f'{v}_pools.npz',transforms=pp,counts=nn,old_counts=oldpool['counts'])
    @lru_cache(maxsize=8)
    def evidence_pool(pe,i,j):
        counts['pair_evidence']+=len(pool[pe,i,j]);return [transport[pe,i,j].evidence(T) for T in pool[pe,i,j]]
    def score_endpoint(key,T):
        i,j,k=key;keys=[(0,i,j),(1,i,k),(2,j,k)];poses=[T[0],T[1],T[1]@inverse(T[0])];ev=[transport[x].evidence(U) for x,U in zip(keys,poses)];counts['pair_evidence']+=3;counts['partition_evaluations']+=1
        a=sum(w*scorers[x].score(U[None])[0] for w,x,U in zip([1.,1.,.5],keys,poses));return np.r_[a,joint_values(ev,cfg)]
    cache={};discrete={};payload={}
    def objective(i,j,k):
        key=i,j,k
        if key not in cache:
            A,B=pool[0,i,j],pool[1,i,k];T1=np.repeat(A,len(B),0);T2=np.tile(B,(len(A),1,1));relative=T2@inverse(T1);ov=ordinary[0,i,j][:,None]+ordinary[1,i,k][None]+.5*scorers[2,j,k].score(relative).reshape(len(A),len(B));h=int(ov.argmin());cc=np.full(4,np.inf);cc[0]=ov.ravel()[h];TT=np.tile(np.array([T1[h],T2[h]]),(4,1,1,1));evA=evidence_pool(0,i,j);evB=evidence_pool(1,i,k)
            # All Cartesian pairs remain eligible. Independent assignment is
            # a valid lower energy bound on both constrained partitions.
            order=np.argsort((paircost[0,i,j][:,None]+paircost[1,i,k][None]).ravel(),kind='stable')
            for h in order:
                a,b=divmod(int(h),len(B));ec=transport[2,j,k].evidence(relative[h]);counts['pair_evidence']+=1;counts['cartesian_pairs']+=1;ev=[evA[a],evB[b],ec];Z=sum(w*r['potential'] for w,r in zip([1.,1.,.5],ev))+cfg['support_regularizer'];free=sum(w*r['free'] for w,r in zip([1.,1.,.5],ev))/2.5;bound=1-sum(w*r['value'] for w,r in zip([1.,1.,.5],ev))/Z+cfg['free_weight']*free
                if bound<cc[1]:cc[1]=bound;TT[1]=[A[a],B[b]]
                if bound>=max(cc[2:]):counts['partition_bound_skips']+=1;continue
                vv=joint_values(ev,cfg);counts['partition_evaluations']+=1
                for m in [2,3]:
                    if vv[m-1]<cc[m]:cc[m]=vv[m-1];TT[m]=[A[a],B[b]]
            cache[key]=[cc,TT,np.full(4,-1,int)];discrete[key]=[cc.copy(),TT.copy()]
            if len(cache)%8==0:progress('full Cartesian joint poses',triples=len(cache),counters=counts)
        return cache[key]
    with np.load(parent/f'{v}_search_graph.npz') as f:initials=list(f['states'])
    costs=[]
    for values in [ordinary,paircost]:
        costs.append({(e,f):np.array([[values[pe,i,j].min() for j in range(24)] for i in range(24)]) for pe,(e,f) in enumerate(PAIRS)})
        a,b=old.solve_assignments(costs[-1],bc);initials.append(np.c_[a,b])
    old_allowed=[[{int(s[i,e]) for s in initials} for i in range(24)] for e in range(2)];proxy=[costs[1][p] for p in PAIRS];allowed=[[old_allowed[e][i]|set(np.argsort(proxy[e][i])[:3].tolist()) for i in range(24)] for e in range(2)]
    states=[];seen=set();values=[]
    def add(s):
        s=np.asarray(s,np.int64)
        if s.tobytes() in seen:return False
        seen.add(s.tobytes());states.append(s.copy());values.append(sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(s)),start=np.zeros(4)));return True
    for s in initials:add(s)
    initial_count=len(states);proposals=[]
    for q in np.unique(np.array(values).argmin(0)):proposals+=legal_neighbors(states[q],allowed,proxy,old_allowed)
    proposals.sort(key=lambda x:(x[0],x[1],x[2].tobytes()));added=0
    for _,__,s in proposals:
        added+=int(add(s))
        if added>=cfg['extra_identity_states']:break
    discrete_values=np.array(values);selected=sorted({(i,int(j),int(k)) for q in np.unique(discrete_values.argmin(0)) for i,(j,k) in enumerate(states[q])})
    progress('continuous poses',triples=len(cache),refine_triples=len(selected));print('TRANSPORT DISCRETE',v,len(cache),round(time.perf_counter()-start,1),flush=True)
    for fi,key in enumerate(selected):
        i,j,k=key;initial=[]
        for T in cache[key][1]:
            if not any(np.max(abs(T-U))<1e-10 for U in initial):initial.append(T.copy())
        rr=[edges[0][i*24+j],edges[1][i*24+k],edges[2][j*24+k]];bb=[banks[0][i],banks[1][j],banks[2][k]];tr=[transport[0,i,j],transport[1,i,k],transport[2,j,k]];n=0
        for T in initial:
            for m,method in enumerate(METHODS):
                if m==0:
                    U,path=ordinary_fit(T,rr,cfg['pose_steps']);clusters=np.zeros((0,3),int)
                else:
                    fitout=fit(T,bb,tr,cfg,method);U,path,clusters=fitout['T'],fitout['path'],fitout['clusters']
                vv=score_endpoint(key,U)
                for name,a in [('initial',T),('T',U),('path',path),('clusters',clusters),('producer',m),('scores',vv)]:payload[f'{fi}_{n}_{name}']=a
                for col in range(4):
                    if vv[col]<cache[key][0][col]:cache[key][0][col]=vv[col];cache[key][1][col]=U;cache[key][2][col]=fi*100+n
                n+=1;counts['fits']+=1
        payload[f'{fi}_count']=n
        if fi%4==0:progress('continuous poses',done=fi,total=len(selected),counters=counts)
    values=np.array([sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(s)),start=np.zeros(4)) for s in states]);states=np.array(states);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_search_graph.npz',states=states,values=values,discrete_values=discrete_values,best=best,initial_count=initial_count)
    np.savez_compressed(root/f'{v}_cache.npz',triples=np.array(list(cache)),costs=np.array([r[0] for r in cache.values()]),transforms=np.array([r[1] for r in cache.values()]),origins=np.array([r[2] for r in cache.values()]),discrete_costs=np.array([r[0] for r in discrete.values()]),discrete_transforms=np.array([r[1] for r in discrete.values()]))
    np.savez_compressed(root/f'{v}_fits.npz',triples=np.array(selected),**payload)
    for col,m in enumerate(METHODS):
        q=best[col];s=states[q];TT=np.array([objective(i,int(j),int(k))[1][col] for i,(j,k) in enumerate(s)]);cc=np.array([objective(i,int(j),int(k))[0][col] for i,(j,k) in enumerate(s)])
        np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=TT,objectives=cc,search_state=q,triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,elapsed_s=time.perf_counter()-start,states=len(states),triples=len(cache),refined_triples=len(selected),counters=counts,private_labels_read=False,same_inputs=True,same_final_candidates=True,equal_flops=False));print('TRANSPORT DONE',v,round(time.perf_counter()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();cp=Path('configs/material_partial_track_transport_v7.json');cfg=json.loads(cp.read_text(encoding='utf-8'))
    if args.profile:
        ic,bc,ep,groups,views,banks,edges=setup(cfg,'rich');pp=[PairTransport(banks[e][0],banks[f][0],edges[pe][0],views[e][0],views[f][0],0,0,cfg) for pe,(e,f) in enumerate(PAIRS)];T=geometry_starts(ep[0][0]['points'],ep[1][0]['points']);start=time.perf_counter()
        for _ in range(4):
            for U in T:joint_values([p.evidence(U) for p in pp],cfg)
        print(json.dumps(dict(public_only=True,per_joint_seconds=(time.perf_counter()-start)/96,nodes=[len(banks[e][0]['xyz']) for e in range(3)])));return
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));parent=Path(cfg['parent_root']).parent;native=Path(ic['native_root']);parentseal=json.loads((parent/'prediction_seal.json').read_text(encoding='utf-8'))
    for p,h in parentseal['sources'].items():assert sha(p)==h
    for p,h in parentseal['files'].items():assert sha(parent/p)==h
    ns=json.loads((native/'correspondence_seal.json').read_text(encoding='utf-8'))
    for p,h in ns['sources'].items():assert sha(p)==h
    for p,h in ns['files'].items():assert sha(native/p)==h
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);paths=[__file__,str(cp),'inspection_decisions/material_partial_track_transport_v7.py','scripts/check_material_partial_track_transport_v7.py',cfg['input_config'],cfg['registration_config']];sources={p:sha(p) for p in paths};start=time.perf_counter()
    save(root/'frozen_protocol.json',dict(config=cfg,methods=METHODS,sources=sources,parent_seal_sha=sha(parent/'prediction_seal.json'),native_seal_sha=sha(native/'correspondence_seal.json'),release_sha=sha(Path(ic['dataset'])/'release_manifest.json'),private_labels_read=False,scope='One TRAIN scene. Exact Cartesian pose scoring on finite common identity states; bounded conditional partition ascent, NOT exact global multiway optimum. All old states plus common new states; continuous union endpoints shared. Same inputs/candidates, not equal FLOPs. No training.'))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,sources=sources,elapsed_s=time.perf_counter()-start,private_labels_read=False,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')}));print('TRANSPORT SEALED',flush=True)

if __name__=='__main__':main()
