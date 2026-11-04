"""Unchanged author PointDSC+official weights on common dense learned matches.

Task adaptation: bidirectional author-threshold consensus fitness and existing
public identity graph. This is a published MODULE control, not 3DMatch results.
"""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.pointdsc_reference_v1 import load_reference,estimate
from scripts.run_material_surface_distribution_v15b import configuration as current_configuration,load,PAIRS
from scripts.prepare_material_surface_fields_v15 import arrays,load_fields
from inspection_decisions.material_surface_distribution_v15b import dense_modes
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import unique_poses,propagation
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.run_textured_material_comparison_v1 import save,sha

METHODS=['pointdsc_top1']


def configuration():
    c=current_configuration();c['output']='results/pointdsc_material_reference_v15';return c


def matched(a,dense,c):
    # Match the official demo's one nearest-feature correspondence per source.
    # Search ALL native targets, with no coarse pose or private positive input.
    target=[]
    for start in range(0,512,64):
        sim=a['desc'][start:start+64]@dense['desc'].T;target.append(sim.argmax(1))
    return [(a['xyz'],dense['xyz'][np.concatenate(target)])]


def job(c,v,profile=False):
    torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path(c['output']);_,_,banks,dense=arrays(c,v);load_fields(c,v,banks,dense);model,cfg=load_reference();start=time.perf_counter();old=load(Path(c['parent_root'])/f'{v}_cache.npz');graph=load(Path(c['parent_root'])/f'{v}_search_graph.npz');payload={};pair_payload={};rows=[];cache={};times=[]
    @lru_cache(maxsize=512)
    def pair(pe,i,j):
        tick=time.perf_counter();e,f=PAIRS[pe];forward=matched(banks[e][i],dense[f][j],c);backward=matched(banks[f][j],dense[e][i],c);poses=[]
        for col in range(len(METHODS)):
            a=estimate(model,*forward[col])[0].astype(float);b=inverse(estimate(model,*backward[col])[0].astype(float));U=unique_poses([a,b]);poses.append(U);pair_payload[f'{pe}_{i}_{j}_{col}']=U
        times.append(time.perf_counter()-tick);return poses,forward,backward
    def score(p,T,col):
        from inspection_decisions.textured_material_matching_v1 import apply
        x,y=p[1][col];b,a=p[2][col];return 1-.5*(np.mean(np.linalg.norm(apply(T,x)-y,axis=1)<cfg['inlier_threshold'])+np.mean(np.linalg.norm(apply(inverse(T),b)-a,axis=1)<cfg['inlier_threshold']))
    triples=old['triples'][:1] if profile else old['triples']
    for q,(i,j,k) in enumerate(triples):
        i,j,k=map(int,(i,j,k));P=[pair(0,i,j),pair(1,i,k),pair(2,j,k)];cost=[];selected=[]
        for col in range(len(METHODS)):
            A,B,C=[p[0][col] for p in P];U=unique_poses(list(np.array([[a,b] for a in A for b in B]))+list(propagation(A,B,C)));V=np.array([score(P[0],T[0],col)+score(P[1],T[1],col)+.5*score(P[2],T[1]@inverse(T[0]),col) for T in U])/2.5;h=int(V.argmin());cost.append(V[h]);selected.append(U[h]);payload[f'{q}_{col}_transforms']=U;payload[f'{q}_{col}_values']=V
        rows.append((np.array(cost),np.array(selected)))
        if profile:return dict(seconds=time.perf_counter()-start,pair_seconds=times,official_config=cfg,private_labels_read=False)
        if q%8==7:save(root/f'{v}_progress.json',dict(triples=q+1,total=len(old['triples']),seconds=time.perf_counter()-start,pairs=len(times)))
    costs=np.array([r[0] for r in rows]);lookup={tuple(t):q for q,t in enumerate(old['triples'])};values=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(len(METHODS))) for s in graph['states']]);best=values.argmin(0)
    np.savez_compressed(root/f'{v}_cache.npz',triples=old['triples'],costs=costs,transforms=np.array([r[1] for r in rows]));np.savez_compressed(root/f'{v}_proposals.npz',**payload);np.savez_compressed(root/f'{v}_pair_proposals.npz',**pair_payload);np.savez_compressed(root/f'{v}_search_graph.npz',states=graph['states'],values=values,best=best)
    for col,m in enumerate(METHODS):
        s=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(s)];np.savez_compressed(root/f'{v}_{m}.npz',target_slots=s,transforms=np.array([rows[q][1][col] for q in ix]),objectives=costs[ix,col],search_state=best[col],triplet_counts=np.zeros(24,int))
    save(root/f'{v}_diagnostics.json',dict(complete=True,private_labels_read=False,seconds=time.perf_counter()-start,pairs=len(times),pair_mean_seconds=float(np.mean(times)),official_config=cfg));print('POINTDSC TASKS DONE',v,round(time.perf_counter()-start,1),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();c=configuration()
    if args.profile:print(json.dumps(job(c,'rich',True),indent=2));return
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False);official=Path('third_party/PointDSC_official_v1');manifest=json.loads((official/'source_manifest.json').read_text(encoding='utf-8'))
    for rec in manifest['files']:assert sha(official/rec['path'])==rec['sha256']
    paths=[__file__,'inspection_decisions/pointdsc_reference_v1.py','scripts/check_pointdsc_reference_v1.py','results/pointdsc_cpu_compatibility_v1.json'];sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=c,methods=METHODS,sources=sources,official_commit=manifest['commit'],official_manifest_sha=sha(official/'source_manifest.json'),field_seal_sha=sha(Path(c['field_cache'])/'field_seal.json'),private_labels_read=False,scope='Unchanged official PointDSC3DMatch weights+default thresholds, shared V3fullnative RGBD descriptors, official-demo one nearest-feature correspondence per source. Task head bidirectional inlier count at official0.1m threshold; fixed V14identitystates, no GT. Not full authors FCGF pipeline, not 3DMatch benchmark, no equalFLOPs/input correspondence budget or shared final pose pool with V15.'))
    with ProcessPoolExecutor(max_workers=3) as executor:
        for f in as_completed([executor.submit(job,c,v) for v in c['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h,p
    save(root/'prediction_seal.json',dict(complete=True,private_labels_read=False,sources=sources,files={p.name:sha(p) for p in root.glob('*.npz')}));print('POINTDSC REFERENCE SEALED',flush=True)


if __name__=='__main__':main()
