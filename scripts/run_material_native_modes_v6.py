"""New continuous proposals from native image matches and common-track modes."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score,legal_neighbors
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges,pair_modes,material_tracks,coupled_modes
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['old_pool','native_ransac','native_graph','joint_all','cycle_independent','shared_modes']
COUPLED=['joint_all','cycle_independent','shared_modes'];PAIRS=[(0,1),(0,2),(1,2)]


class NativeScore:
    def __init__(self,ordinary,edge):
        self.ordinary=ordinary;self.edge=edge
        _,self.a=np.unique(edge['source'],return_inverse=True);_,self.b=np.unique(edge['target'],return_inverse=True)
    def score(self,poses):
        result=ordinary_score(self.ordinary,poses);r=self.edge
        if not len(r['weight']):return result+1
        out=[]
        for start in range(0,len(poses),128):
            T=poses[start:start+128];x=np.einsum('bij,nj->bni',T[:,:3,:3],r['x'])+T[:,None,:3,3];d2=((x-r['y'][None])**2).sum(2)
            values=np.exp(-.5*d2/.025**2)*r['weight'][None];a=np.zeros((len(T),self.a.max()+1));b=np.zeros((len(T),self.b.max()+1))
            np.maximum.at(a.T,self.a,values.T);np.maximum.at(b.T,self.b,values.T)
            support=np.minimum(a.sum(1),b.sum(1));out.append(1-np.minimum(1,support/6))
        return result+np.concatenate(out)


def merge_poses(base,ras,graph):
    values=list(base);keys={np.round(T,8).tobytes():h for h,T in enumerate(values)};masks=[set(range(len(base))) for _ in range(3)]
    for family,extra in [(1,ras),(2,graph)]:
        for T in extra:
            if T is None:continue
            key=np.round(T,8).tobytes()
            if key not in keys:keys[key]=len(values);values.append(T)
            masks[family].add(keys[key])
    mask=np.zeros((3,len(values)),bool)
    for h,ii in enumerate(masks):mask[h,list(ii)]=True
    return np.array(values),mask


def common_search(initials,objective,allowed,cost,old_allowed):
    states=[];values=[];seen=set();paths=[]
    def add(p):
        p=np.asarray(p,dtype=np.int64);key=p.tobytes()
        if key in seen:return
        seen.add(key);states.append(p.copy());values.append(sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(p)),start=np.zeros(6)))
    for p in initials:add(p)
    initial=len(states)
    for _ in range(2):
        v=np.array(values);proposals=[]
        for k in np.unique(v.argmin(0)):proposals.extend(legal_neighbors(states[k],allowed,cost,old_allowed))
        proposals.sort(key=lambda a:(a[0],a[1],a[2].tobytes()));added=0
        for _,__,p in proposals:
            if np.asarray(p,dtype=np.int64).tobytes() in seen:continue
            add(p);added+=1
            if added>=64:break
        paths.append(np.array(values).min(0))
        if not added:break
    values=np.array(values);return dict(states=np.array(states),values=values,best=values.argmin(0),paths=np.array(paths),initial=initial)


def generate(name,cfg,base,output):
    torch.set_num_threads(1);root=Path(output)/name;root.mkdir();start=time.perf_counter();native=Path(cfg['native_root'])/name;parent=Path(cfg['parent_root'])/name
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    def progress(message,**extra):
        save(root/'progress.json',dict(message=message,elapsed_s=time.perf_counter()-start,**extra));print('V6 MODES',name,message,round(time.perf_counter()-start,1),flush=True)
    for v in base['variants']:
        frames=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[0] for e in range(3)]
        lf=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)];ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
        with np.load(native/f'{v}_native_edges.npz') as f:edges=unpack_edges(dict(f),cfg)
        with np.load(parent/f'{v}_union_pools.npz') as f:oldpool=dict(f)
        pools={};masks={};pair_graph={};scorers={};edge_value={};raw_pair_counts=[]
        for pe,(e,f) in enumerate(PAIRS):
            pools[e,f]=[];masks[e,f]=[];pair_graph[e,f]=[]
            for i in range(24):
                pr=[];mr=[];gr=[]
                for j in range(24):
                    record=edges[pe][i*24+j];ras=pair_modes(record,cfg,'ransac',cfg['seed']+pe*10000+i*24+j);graph=pair_modes(record,cfg,'graph',0);gr.append(graph)
                    ras=ras+[old.icp(ep[e][i],ep[f][j],T,bcfg) for T in ras];gg=graph+[old.icp(ep[e][i],ep[f][j],T,bcfg) for T in graph]
                    parentposes=oldpool['transforms'][pe,i,j,:oldpool['counts'][pe,i,j]];pp,mm=merge_poses(parentposes,ras,gg);pr.append(pp);mr.append(mm)
                    scorer=NativeScore(PairScore(ep[e][i],ep[f][j],old.matches(ep[e][i],ep[f][j],bcfg),groups[e][i],groups[f][j],{},frames[f],j,base),record)
                    scorers[e,f,i,j]=scorer;edge_value[e,f,i,j]=scorer.score(pp);raw_pair_counts.append((len(record['weight']),len(ras),len(gg)))
                pools[e,f].append(pr);masks[e,f].append(mr);pair_graph[e,f].append(gr)
            progress(v+' native pair modes '+str((e,f)))
        tracks,rawtracks=material_tracks(edges,cfg);couples={m:{} for m in COUPLED};keys=[];payload={};fit_total=0
        for key,track in sorted(tracks.items()):
            if len(track['x'])<3:continue
            i,j,k=key;keys.append(key);q=len(keys)-1
            for field in ['x','node','weight']:payload[f'{q}_{field}']=track[field]
            for m in COUPLED:
                rr=coupled_modes(track,pair_graph[0,1][i][j],pair_graph[0,2][i][k],cfg,m);couples[m][key]=rr;fit_total+=len(rr)
                for h,r in enumerate(rr):
                    for field in ['T','initial','path','selected']:payload[f'{q}_{m}_{h}_{field}']=r[field]
                payload[f'{q}_{m}_count']=len(rr)
        payload['keys']=np.array(keys,int).reshape(-1,3);np.savez_compressed(root/f'{v}_material_modes.npz',**payload)
        cap=max(len(p) for mat in pools.values() for row in mat for p in row);packed=np.tile(np.eye(4),(3,24,24,cap,1,1));counts=np.zeros((3,24,24),int);allowed_masks=np.zeros((3,3,24,24,cap),bool)
        for pe,key in enumerate(PAIRS):
            for i in range(24):
                for j in range(24):
                    pp=pools[key][i][j];counts[pe,i,j]=len(pp);packed[pe,i,j,:len(pp)]=pp;allowed_masks[:,pe,i,j,:len(pp)]=masks[key][i][j]
        np.savez_compressed(root/f'{v}_pair_pools.npz',transforms=packed,counts=counts,family_masks=allowed_masks)
        progress(v+' actual material tracks',tuples=len(keys),raw_tracks=sum(rawtracks.values()),fits=fit_total)
        cache={};calls=0
        def objective(i,j,k):
            nonlocal calls
            key=i,j,k
            if key not in cache:
                A,B=pools[0,1][i][j],pools[0,2][i][k];T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1))
                values=edge_value[0,1,i,j][:,None]+edge_value[0,2,i,k][None,:]+.5*scorers[1,2,j,k].score(T2@inverse(T1)).reshape(len(A),len(B))
                costs=[];poses=[];origins=[]
                for family in range(3):
                    ok=masks[0,1][i][j][family,:,None]&masks[0,2][i][k][family,None,:];vv=np.where(ok,values,np.inf);h=int(vv.argmin());costs.append(float(vv.ravel()[h]));poses.append([T1[h],T2[h]]);origins.append(-1)
                for m in COUPLED:
                    best=costs[2];TT=poses[2];origin=-1;rr=couples[m].get(key,[])
                    if rr:
                        G=np.array([r['T'] for r in rr]);vv=scorers[0,1,i,j].score(G[:,0])+scorers[0,2,i,k].score(G[:,1])+.5*scorers[1,2,j,k].score(G[:,1]@inverse(G[:,0]));h=int(vv.argmin())
                        if vv[h]<best:best=float(vv[h]);TT=G[h];origin=h
                    costs.append(best);poses.append(TT);origins.append(origin)
                cache[key]=(np.array(costs),np.array(poses),np.array(origins));calls+=1
                if calls%150==0:progress(v+' full joint hypotheses',triples=calls)
            return cache[key]
        common_cost=[]
        for e,f in PAIRS:common_cost.append(np.array([[edge_value[e,f,i,j].min() for j in range(24)] for i in range(24)]))
        initials=[]
        for m in ['baseline','ncc','random','dense','surface','geo','union']:
            with np.load(parent/f'{v}_{m}.npz') as f:initials.append(f['target_slots'])
        for m in ['ordinary_joint','fusion','independent','shared']:
            with np.load(Path('results/material_joint_shared_comparison_v5')/name/f'{v}_{m}.npz') as f:initials.append(f['target_slots'])
        for family in [1,2]:
            cost={key:np.array([[edge_value[*key,i,j][masks[key][i][j][family]].min() for j in range(24)] for i in range(24)]) for key in PAIRS}
            a,b=old.solve_assignments(cost,bcfg);initials.append(np.c_[a,b])
        old_allowed=[[{int(s[i,e]) for s in initials} for i in range(24)] for e in [0,1]]
        allowed=[[old_allowed[e][i]|set(np.argsort(common_cost[e][i])[:3].tolist()) for i in range(24)] for e in [0,1]]
        for i,j,k in keys:allowed[0][i].add(j);allowed[1][i].add(k)
        graph=common_search(initials,objective,allowed,common_cost,old_allowed)
        np.savez_compressed(root/f'{v}_search_graph.npz',**graph)
        kk=np.array(list(cache),int);np.savez_compressed(root/f'{v}_joint_cache.npz',triples=kk,costs=np.array([cache[tuple(k)][0] for k in kk]),transforms=np.array([cache[tuple(k)][1] for k in kk]),origins=np.array([cache[tuple(k)][2] for k in kk]))
        for col,m in enumerate(METHODS):
            state=int(graph['best'][col]);slots=graph['states'][state];TT=np.array([objective(i,int(j),int(k))[1][col] for i,(j,k) in enumerate(slots)]);vv=np.array([objective(i,int(j),int(k))[0][col] for i,(j,k) in enumerate(slots)])
            np.savez_compressed(root/f'{v}_{m}.npz',target_slots=slots,transforms=TT,objectives=vv,triplet_counts=np.zeros(24,int),search_state=state)
        save(root/f'{v}_diagnostics.json',dict(pair_counts=np.array(raw_pair_counts),raw_track_tuples=len(tracks),processed_track_tuples=len(keys),raw_tracks=sum(rawtracks.values()),track_keys=keys,
            fits={m:sum(len(a) for a in vv.values()) for m,vv in couples.items()},search_states=len(graph['states']),triples=len(cache),no_identity_topk_pruning=True,
            note='All old poses retained. Pair family masks differ, joint modes remain COUPLED per identity tuple; never expanded into an uncontrolled Cartesian mixture. Same native-fusion score and common finite identity graph.'))
        progress(v+' COMPLETE')
    progress('COMPLETE')


def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));base=json.loads(Path(cfg['base_config']).read_text(encoding='utf-8'));native=Path(cfg['native_root']);start=time.monotonic()
    while True:
        try:s=json.loads((native/'correspondence_seal.json').read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):s={}
        if s.get('complete'):break
        if time.monotonic()-start>3600:raise TimeoutError('Native correspondence seal not ready; no inference killed')
        time.sleep(2)
    for p,h in s['sources'].items():assert sha(p)==h
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);proto=json.loads((native/'frozen_protocol.json').read_text(encoding='utf-8'));scenes=proto['scenes']
    sources={p:sha(p) for p in [__file__,'configs/material_native_modes_v6.json','inspection_decisions/material_correspondence_mode_graph_v6.py','inspection_decisions/material_joint_id_pose_search_v5.py','inspection_decisions/material_correspondence_inference_v3.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,base_config=base,legacy_config=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8')),methods=METHODS,scenes=scenes,sources=sources,
        native_seal_sha=sha(native/'correspondence_seal.json'),parent_seal_sha=sha(Path(cfg['parent_root'])/'prediction_seal.json'),dataset_release_sha=sha(Path(base['dataset'])/'release_manifest.json'),
        native_score='Exact old geometry/SIFT/ray + uniqueness-capped native confidence support (sigma25mm, support6), identical for all methods',
        search='common initial hypotheses +2 rounds x64 neighbors across all six objectives, all pose pairs of each family + exact coupled mode hypotheses',
        evaluation='four reused DEV scenes, not independent confirmation',controls=[dict(baseline='old_pool',candidate='native_ransac'),dict(baseline='native_ransac',candidate='native_graph'),dict(baseline='native_graph',candidate='joint_all'),dict(baseline='joint_all',candidate='shared_modes'),dict(baseline='cycle_independent',candidate='shared_modes')]))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(generate,name,cfg,base,str(root)) for name in scenes]):f.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('V6 MODE PREDICTIONS SEALED',flush=True)


if __name__=='__main__':main()
