"""Fixed-budget minority-consensus intervention, frozen public fields only."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import argparse,json,time,py_compile
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.run_material_surface_distribution_v15b import configuration as parent_configuration,PAIRS,load,CachedPair,Triple
from scripts.prepare_material_surface_fields_v15 import arrays,load_fields
from inspection_decisions.material_continuous_surface_cache_v14b import DensePair
from inspection_decisions.material_continuous_exposure_v14c import directed_match
from inspection_decisions.material_surface_distribution_v15b import dense_modes,RegionPair
from inspection_decisions.material_partial_consensus_graph_v16 import proposals,FAMILIES,inverse
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import unique_poses,propagation
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['retained_v15_point','common_geometry','common_descriptor_fusion','common_point_density','family_global','family_local','family_triplet','family_partial']
SCORE_COLUMNS=np.array([0,2,3])


def configuration():
    c=parent_configuration();c.update(json.loads(Path('configs/material_partial_consensus_graph_v16.json').read_text(encoding='utf-8')));return c


def job(c,v,profile=False):
    torch.set_num_threads(1);torch.set_flush_denormal(True);start=time.perf_counter();root=Path(c['output']);frames,views,banks,dense=arrays(c,v);load_fields(c,v,banks,dense);parent=Path(c['parent_root']);old=load(parent/f'{v}_cache.npz');oldU=load(parent/f'{v}_proposals.npz');graph=load(parent/f'{v}_search_graph.npz');payload={};pair_payload={};rows=[];pairtimes=[];statistics={f:0 for f in FAMILIES};statistics['pair_count']=0
    @lru_cache(maxsize=512)
    def pair(pe,i,j):
        tick=time.perf_counter();e,f=PAIRS[pe];a={**banks[e][i],'normalizer':banks[e][i][f'normalizer_{f}']};b={**banks[f][j],'normalizer':banks[f][j][f'normalizer_{e}']};base=DensePair(a,b,views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],c);fm=dense_modes(a,dense[f][j],c);bm=dense_modes(b,dense[e][i],c);sc=CachedPair(RegionPair(base,fm,bm,c));sc.cache={};forward,ft=proposals(a,fm,dense[f][j],c);backward,bt=proposals(b,bm,dense[e][i],c);selected=[]
        for direction,tr in [('f',ft),('b',bt)]:
            for name,value in tr.items():pair_payload[f'{pe}_{i}_{j}_{direction}_{name}']=value
        for family in FAMILIES:
            raw=unique_poses(list(forward[family])+list(inverse(backward[family])));statistics[family]+=len(raw);pair_payload[f'{pe}_{i}_{j}_{family}_raw']=raw
            # Every generator uses the SAME public point-density screening.
            val=np.array([sc.score(T)[3] for T in raw]);poses=raw[np.argsort(val,kind='stable')[:c['global_keep']]];selected.append(poses);pair_payload[f'{pe}_{i}_{j}_{family}']=poses
        statistics['pair_count']+=1;pairtimes.append(time.perf_counter()-tick);return sc,selected
    for q,(i,j,k) in enumerate(old['triples'][:1 if profile else None]):
        i,j,k=map(int,(i,j,k));P=[pair(0,i,j),pair(1,i,k),pair(2,j,k)];sc=Triple([p[0] for p in P]);families=[];old_best=old['transforms'][q,5]
        for family in range(4):
            A,B,C=[p[1][family] for p in P];extra=[[a,b] for a in A for b in B]+list(propagation(A[:2],B[:2],C[:2]));extra += [[a,old_best[1]] for a in A]+[[old_best[0],b] for b in B];families.append(extra)
        retained=oldU[f'{q}_transforms'];U=unique_poses(list(retained)+sum(families,[]));eligible=np.zeros((len(U),4),bool);eligible[:len(retained)]=True
        for h,T in enumerate(U[len(retained):],start=len(retained)):
            for f in range(4):eligible[h,f]=any(np.max(abs(T-Q))<1e-9 for Q in families[f])
        tick=time.perf_counter();values=np.r_[oldU[f'{q}_values'][:,SCORE_COLUMNS],sc.score(U[len(retained):])[:,SCORE_COLUMNS]] if len(U)>len(retained) else oldU[f'{q}_values'][:,SCORE_COLUMNS].copy();best=values.argmin(0);fbest=[np.flatnonzero(eligible[:,f])[np.argmin(values[eligible[:,f],2])] for f in range(4)];cost=np.r_[old['costs'][q,5],values[best,np.arange(3)],values[fbest,2]];T=np.r_[old['transforms'][q,5:6],U[best],U[fbest]];rows.append((cost,T));payload[f'{q}_transforms']=U;payload[f'{q}_values']=values;payload[f'{q}_family_eligible']=eligible;payload[f'{q}_retained_count']=np.array(len(retained))
        if profile:return dict(seconds=time.perf_counter()-start,pair_seconds=pairtimes,score_seconds=time.perf_counter()-tick,candidates=len(U),retained=len(retained),values=cost.tolist(),statistics=statistics,private_labels_read=False)
        if q%8==7 or q==len(old['triples'])-1:
            save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,statistics=statistics))
        if time.perf_counter()-start>c['hard_timeout_s']:raise TimeoutError(v)
    costs=np.array([r[0] for r in rows]);lookup={tuple(t):q for q,t in enumerate(old['triples'])};svalues=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8)) for s in graph['states']]);best=svalues.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=costs,transforms=np.array([r[1] for r in rows]));np.savez_compressed(root/f'{v}_proposals.npz',**payload);np.savez_compressed(root/f'{v}_pair_proposals.npz',**pair_payload);np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=svalues,best=best)
    for col,m in enumerate(METHODS):
        s=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(s)];np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=np.array([rows[q][1][col] for q in ix]),objectives=costs[ix,col],search_state=best[col],triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,private_labels_read=False,seconds=time.perf_counter()-start,statistics=statistics,triples=len(rows),states=len(graph['states']),pair_seconds_mean=float(np.mean(pairtimes)),source_centers=36864,native_pixels=sum(len(a['xyz']) for row in dense for a in row)));print('V16 DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();c=configuration();field=Path(c['field_cache']);fs=json.loads((field/'field_seal.json').read_text(encoding='utf-8'));assert fs['complete'] and not fs['private_labels_read']
    for p,h in fs['sources'].items():assert sha(p)==h,p
    for p,h in fs['files'].items():assert sha(field/p)==h,p
    if args.profile:print(json.dumps(job(c,'rich',True),indent=2));return
    parent=Path(c['parent_root']);seal=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in seal[group].items():assert sha(p)==h,p
    paths=[__file__,'scripts/score_material_partial_consensus_v16.py','inspection_decisions/material_partial_consensus_graph_v16.py','configs/material_partial_consensus_graph_v16.json','scripts/check_material_partial_consensus_v16.py','results/material_partial_consensus_checks_v16.json']
    for p in paths:
        if p.endswith('.py'):py_compile.compile(p,doraise=True)
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False);sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=c,methods=METHODS,sources=sources,field_seal_sha=sha(field/'field_seal.json'),parent_completion_sha=sha(parent/'completion_integrity.json'),private_labels_read=False,new_training=False));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=c['workers']) as executor:
        for f in as_completed([executor.submit(job,c,v) for v in c['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h,p
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,new_training=False,sources=sources,files={p.name:sha(p) for p in root.glob('*.npz')},seconds=time.perf_counter()-start));print('V16 SEALED',flush=True)


if __name__=='__main__':main()
