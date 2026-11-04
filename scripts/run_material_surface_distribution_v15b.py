"""Full24 learned dense-region global proposals, common actual endpoint pool."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse,py_compile
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scripts.prepare_material_surface_fields_v15 import configuration as field_configuration,arrays,load_fields
from scripts.run_material_continuous_surface_correspondence_v14c import load,dense_icp,PAIRS
from inspection_decisions.material_continuous_surface_cache_v14b import DensePair
from inspection_decisions.material_continuous_exposure_v14c import directed_match
from inspection_decisions.material_surface_distribution_v15b import dense_modes,global_proposals,RegionPair
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import unique_poses,propagation
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['retained_v14_geometry','retained_v14_continuous','common_geometry','common_rgb_fusion','common_descriptor_fusion','common_point_density','common_mean_density','common_region_density']


def configuration():
    c=field_configuration();c.update(json.loads(Path('configs/material_surface_distribution_v15b.json').read_text(encoding='utf-8')));c['output']=c['prediction_output'];return c


class CachedPair:
    def __init__(self,region):self.region=region;self.cache={}
    def score(self,T):
        key=np.asarray(T,np.float64).tobytes()
        if key not in self.cache:
            if len(self.cache)>2048:self.cache.clear()
            self.cache[key]=self.region.values(T)
        return self.cache[key]


class Triple:
    def __init__(self,pairs):self.pairs=pairs
    def score(self,U):
        from inspection_decisions.material_surface_patch_registration_v9 import inverse
        return np.array([(self.pairs[0].score(T[0])+self.pairs[1].score(T[1])+.5*self.pairs[2].score(T[1]@inverse(T[0])))/2.5 for T in U])


def job(c,v,profile=False):
    torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path(c['output']);start=time.perf_counter();frames,views,banks,dense=arrays(c,v);load_fields(c,v,banks,dense);parent=Path(c['parent_root']);old=load(parent/f'{v}_cache.npz');oldU=load(parent/f'{v}_proposals.npz');graph=load(parent/f'{v}_search_graph.npz');payload={};pair_payload={};rows=[];statistics=dict(dense_mode_families=0,point_models=0,region_models=0,icp_fits=0);pairtimes=[]
    @lru_cache(maxsize=512)
    def pair(pe,i,j):
        tick=time.perf_counter();e,f=PAIRS[pe];a={**banks[e][i],'normalizer':banks[e][i][f'normalizer_{f}']};b={**banks[f][j],'normalizer':banks[f][j][f'normalizer_{e}']};base=DensePair(a,b,views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],c);forward=dense_modes(a,dense[f][j],c);backward=dense_modes(b,dense[e][i],c);statistics['dense_mode_families']+=2;rp=RegionPair(base,forward,backward,c);sc=CachedPair(rp);point=list(global_proposals(a,forward,c,False));region=list(global_proposals(a,forward,c,True))
        from inspection_decisions.material_surface_patch_registration_v9 import inverse
        point+=list(inverse(global_proposals(b,backward,c,False)));region+=list(inverse(global_proposals(b,backward,c,True)));statistics['point_models']+=len(point);statistics['region_models']+=len(region);selected=[]
        for family,poses,column in [('point',point,2),('region',region,5)]:
            poses=unique_poses(poses);pair_payload[f'{pe}_{i}_{j}_{family}_raw']=poses;val=np.array([sc.score(T)[column] for T in poses]);chosen=poses[np.argsort(val,kind='stable')[:c['global_keep']]]
            if len(chosen):chosen=unique_poses(list(chosen)+[dense_icp(base,chosen[0],c,False)]);statistics['icp_fits']+=1
            selected.append(chosen);pair_payload[f'{pe}_{i}_{j}_{family}']=chosen
        pairtimes.append(time.perf_counter()-tick);return sc,selected
    count=1 if profile else len(old['triples'])
    for q,(i,j,k) in enumerate(old['triples'][:count]):
        i,j,k=map(int,(i,j,k));P=[pair(0,i,j),pair(1,i,k),pair(2,j,k)];sc=Triple([p[0] for p in P]);families=[]
        for family in range(2):
            A,B,C=[p[1][family] for p in P];families.append(list(np.array([[a,b] for a in A for b in B]))+list(propagation(A[:2],B[:2],C[:2])))
        A,B,C=[unique_poses(list(p[1][0])+list(p[1][1])) for p in P];extra=list(np.array([[a,b] for a in A for b in B]))+list(propagation(A[:2],B[:2],C[:2]));retained=oldU[f'{q}_transforms'];U=unique_poses(list(retained)+families[0]+families[1]+extra);eligibility=np.zeros((len(U),2),bool);stage=np.full(len(U),4,int);stage[:len(retained)]=0
        for h,T in enumerate(U):
            for family in range(2):eligibility[h,family]=h<len(retained) or any(np.max(abs(T-Q))<1e-9 for Q in families[family])
            if h>=len(retained):stage[h]=3 if eligibility[h].all() else 1 if eligibility[h,0] else 2 if eligibility[h,1] else 4
        tick=time.perf_counter();values=sc.score(U);best=values.argmin(0);cost=np.r_[old['costs'][q,[1,4]],values[best,np.arange(6)]];T=np.r_[old['transforms'][q,[1,4]],U[best]];rows.append((cost,T));payload[f'{q}_transforms']=U;payload[f'{q}_values']=values;payload[f'{q}_family_eligible']=eligibility;payload[f'{q}_stage']=stage
        if profile:return dict(seconds=time.perf_counter()-start,pair_seconds=pairtimes,score_seconds=time.perf_counter()-tick,candidates=len(U),values=cost.tolist(),private_labels_read=False)
        if q%8==7:
            np.savez_compressed(root/f'{v}_block_{q-7:04}.npz',triples=old['triples'][q-7:q+1],costs=np.array([r[0] for r in rows[-8:]]),transforms=np.array([r[1] for r in rows[-8:]]),**{key:val for key,val in payload.items() if q-7<=int(key.split('_')[0])<=q});save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,statistics=statistics))
        if time.perf_counter()-start>c['hard_timeout_s']:raise TimeoutError(v)
    costs=np.array([r[0] for r in rows]);lookup={tuple(t):q for q,t in enumerate(old['triples'])};svalues=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(8)) for s in graph['states']]);best=svalues.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=costs,transforms=np.array([r[1] for r in rows]));np.savez_compressed(root/f'{v}_proposals.npz',**payload);np.savez_compressed(root/f'{v}_pair_proposals.npz',**pair_payload);np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=svalues,best=best)
    for col,m in enumerate(METHODS):
        s=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(s)];np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=np.array([rows[q][1][col] for q in ix]),objectives=costs[ix,col],search_state=best[col],triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,private_labels_read=False,seconds=time.perf_counter()-start,statistics=statistics,triples=len(rows),states=len(graph['states']),pair_seconds_mean=float(np.mean(pairtimes)),source_centers=36864,native_pixels=sum(len(a['xyz']) for row in dense for a in row)));print('V15b DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();c=configuration();field=Path(c['field_cache']);fs=json.loads((field/'field_seal.json').read_text(encoding='utf-8'));assert fs['complete'] and not fs['private_labels_read']
    for p,h in fs['sources'].items():assert sha(p)==h,p
    for p,h in fs['files'].items():assert sha(field/p)==h,p
    if args.profile:print(json.dumps(job(c,'rich',True),indent=2));return
    parent=Path(c['parent_root']);seal=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in seal[group].items():assert sha(p)==h,p
    paths=[__file__,'scripts/score_material_surface_distribution_v15b.py','inspection_decisions/material_surface_distribution_v15b.py','configs/material_surface_distribution_v15b.json','configs/material_surface_distribution_matching_v15.json','scripts/check_material_surface_regions_v15b.py','scripts/check_material_surface_denormals_v15.py','results/material_surface_denormal_checks_v15.json']
    for p in paths:
        if str(p).endswith('.py'):py_compile.compile(p,doraise=True)
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False);sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=c,methods=METHODS,sources=sources,field_seal_sha=sha(field/'field_seal.json'),parent_completion_sha=sha(parent/'completion_integrity.json'),scope=c['contract']+' '+c['region_contract'],private_labels_read=False,new_training=False));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        for f in as_completed([pool.submit(job,c,v) for v in c['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h,p
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,new_training=False,sources=sources,files={p.name:sha(p) for p in root.glob('*.npz')},seconds=time.perf_counter()-start));print('V15b SEALED',flush=True)


if __name__=='__main__':main()
