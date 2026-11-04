"""All-identity proposal ablation under ONE frozen ordinary ray scorer."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse,unique_poses
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher,surface_grids,decode_local,closest_on_triangles,compatibility_proposals
from scripts.run_material_correspondence_v3 import assignments
from scripts.run_textured_material_comparison_v1 import save,sha


FAMILIES=['ncc','random','dense','surface','geo']
METHODS=['baseline']+FAMILIES+['union']
PAIRS=[(0,1),(0,2),(1,2)]


def pose_arrays(pools):
    cap=max(len(p) for mat in pools.values() for row in mat for p in row)
    packed=np.tile(np.eye(4),(3,24,24,cap,1,1));counts=np.zeros((3,24,24),int)
    for pe,key in enumerate(PAIRS):
        for i in range(24):
            for j in range(24):p=pools[key][i][j];counts[pe,i,j]=len(p);packed[pe,i,j,:len(p)]=p
    return dict(transforms=packed,counts=counts)


def encode_fields(groups,model,random):
    out=[]
    with torch.inference_mode():
        for gg in groups:
            ep=[]
            for g in gg:
                p=torch.from_numpy(g['patch']);m=model.encode(p);r=random.encode(p)
                n=F.normalize(F.unfold(p,3,padding=1).transpose(1,2),dim=2).reshape(len(p),12,12,-1).permute(0,3,1,2)
                ep.append(dict(dense=m.flatten(2).transpose(1,2).numpy(),dense_query=model.query(m).numpy(),
                    random=r.flatten(2).transpose(1,2).numpy(),random_query=random.query(r).numpy(),
                    ncc=n.flatten(2).transpose(1,2).numpy(),ncc_query=F.normalize(n[:,:,5:7,5:7].mean((2,3)),dim=1).numpy()))
            out.append(ep)
    return out


def local_matches(a,b,coarse_a,coarse_b,field_a,field_b,grid,cfg):
    # Fixed source query budget, selected without identity or private labels.
    ii=np.linspace(0,len(a['xyz'])-1,min(cfg['proposal_queries'],len(a['xyz'])),dtype=int)
    anchors=np.argsort(-(coarse_a[ii]@coarse_b.T),axis=1)[:,:cfg['coarse_candidates']]
    ids=np.repeat(ii,anchors.shape[1]);anchors=anchors.ravel();row=np.repeat(np.arange(len(ii)),len(anchors)//len(ii))
    answer={}
    for mode in ['ncc','random','dense']:
        source=field_a[mode+'_query'][ii];target=field_b[mode][anchors]
        scores=np.einsum('id,ivd->iv',source[row],target)/.12
        valid=grid['valid'][anchors];logits=np.where(valid,scores,-1e4);peak=logits.argmax(1)
        y=grid['xyz'][anchors,peak];okay=valid.any(1)
        # Confidence from conditional local distribution; no DEV threshold or
        # uncalibrated null-head hard rejection. All coarse modes survive.
        prob=np.exp(logits-logits.max(1,keepdims=True));prob*=valid;conf=prob.max(1)/np.maximum(prob.sum(1),1e-12)
        answer[mode]=(a['xyz'][ids[okay]],y[okay],conf[okay],ids[okay])
        if mode=='dense':
            continuous=[]
            for score,anchor in zip(scores,anchors):
                point,_=decode_local(score,grid,int(anchor),True)
                continuous.append(np.zeros(3) if point is None else point)
            answer['surface']=(a['xyz'][ids[okay]],np.array(continuous)[okay],conf[okay],ids[okay])
    return answer


def geometric_surface(T,a,grid,steps=6):
    points=grid['xyz'][grid['valid']]
    if len(points)<3:return T
    owner=np.broadcast_to(np.arange(len(grid['xyz']))[:,None],grid['valid'].shape)[grid['valid']];tree=cKDTree(points)
    triangles=grid['triangles'];cur=T.copy()
    for _ in range(steps):
        x=old.apply(cur,a);distance,ix=tree.query(x);targets=[]
        for p,k,observed in zip(x,owner[ix],points[ix]):
            tri=grid['xyz'][k,triangles[grid['tri_valid'][k]]];q=closest_on_triangles(p,tri)
            targets.append(observed if q is None else q)
        targets=np.array(targets);weight=1/(1+(distance/.04)**4);weight[distance>.15]=0
        if (weight>0).sum()<3:break
        value=old.rigid(a,targets,weight)
        if value is None:break
        if np.max(abs(value-cur))<1e-8:cur=value;break
        cur=value
    return cur


def select(pools,scores,bcfg,legacy):
    costs={key:np.array([[scores[key][i][j].min() for j in range(24)] for i in range(24)]) for key in PAIRS}
    p1,p2=assignments(costs,bcfg,legacy);selected=[];energies=[]
    # Pair scorers supplied by caller through score closure (not GT).
    return p1,p2,selected,energies


def job(name,cfg,v4,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(output)/name;root.mkdir();start=time.perf_counter()
    def progress(message):
        save(root/'progress.json',dict(message=message,elapsed_s=time.perf_counter()-start));print('V4',name,message,round(time.perf_counter()-start,1),flush=True)
    coarse=CorrespondenceModel();coarse.load_state_dict(torch.load(Path(cfg['training_root'])/'soft3_31.pt',weights_only=True,map_location='cpu')['state_dict']);coarse.eval()
    model=SurfaceMatcher();model.load_state_dict(torch.load(Path(v4['training_root'])/'matcher_31.pt',weights_only=True,map_location='cpu')['state_dict']);model.eval()
    random=SurfaceMatcher();random.load_state_dict(torch.load(Path(v4['training_root'])/'untrained_31.pt',weights_only=True,map_location='cpu')['state_dict']);random.eval()
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'));oldroot=Path('results/textured_material_comparison_v1')/name
    v3root=Path(cfg['comparison_root'])/name
    for variant in cfg['variants']:
        frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        legacy_frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(legacy_frames)];groups=[observed_groups(f,cfg) for f in frames]
        grids=[surface_grids(f,g,cfg,v4) for f,g in zip(frames,groups)]
        with torch.inference_mode():enc=[[coarse.encode(torch.from_numpy(g['patch'])).numpy() for g in gg] for gg in groups]
        fields=encode_fields(groups,model,random);legacy={}
        for method in ['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
            with np.load(oldroot/f'{variant}_{method}.npz') as f:legacy[method]=dict(f)
        with np.load(v3root/f'{variant}_ray.npz') as f:legacy['v3_ray']=dict(f)
        with np.load(v3root/f'{variant}_union_pools.npz') as f:original=dict(f)
        pools={m:{} for m in METHODS};values={m:{} for m in METHODS};scorers={};field_data={m:np.full((3,24,24,96,3),np.nan,np.float32) for m in ['ncc','random','dense','surface']}
        source_indices=np.full((3,24,24,96),-1,np.int16)
        for pe,(e,f) in enumerate(PAIRS):
            scorers[e,f]=[]
            for m in METHODS:pools[m][e,f]=[];values[m][e,f]=[]
            for i in range(24):
                pr={m:[] for m in METHODS};vr={m:[] for m in METHODS};sr=[]
                for j in range(24):
                    a,b=ep[e][i],ep[f][j];mm=old.matches(a,b,bcfg);ga,gb=groups[e][i],groups[f][j]
                    scorer=PairScore(a,b,mm,ga,gb,{},frames[f],j,cfg);base=original['transforms'][pe,i,j,:original['counts'][pe,i,j]]
                    pred=legacy['v3_ray'];slots=np.c_[np.arange(24),pred['target_slots']];GG=np.tile(np.eye(4),(24,3,1,1));GG[:,1:]=pred['transforms']
                    hh=np.flatnonzero((slots[:,e]==i)&(slots[:,f]==j))
                    base=unique_poses(list(base)+[GG[h,f]@inverse(GG[h,e]) for h in hh])
                    bv=scorer.score(base,['ray'])['ray'];matches=local_matches(ga,gb,enc[e][i],enc[f][j],fields[e][i],fields[f][j],grids[f][j],v4)
                    additions={}
                    for m in ['ncc','random','dense','surface']:
                        x,y,w,ids=matches[m];field_data[m][pe,i,j,:len(y)]=y;source_indices[pe,i,j,:len(ids)]=ids
                        added=compatibility_proposals(x,y,w,ids,v4['proposal_seeds']);additions[m]=unique_poses(added+[old.icp(a,b,T,bcfg) for T in added])
                    query=ga['xyz'][np.linspace(0,len(ga['xyz'])-1,min(v4['proposal_queries'],len(ga['xyz'])),dtype=int)]
                    additions['geo']=unique_poses([geometric_surface(base[k],query,grids[f][j]) for k in np.argsort(bv)[:v4['proposal_seeds']]])
                    pp={'baseline':base}
                    for m in FAMILIES:pp[m]=unique_poses(list(base)+list(additions[m]))
                    pp['union']=unique_poses(list(base)+[T for v in additions.values() for T in v])
                    for m in METHODS:
                        pr[m].append(pp[m]);vr[m].append(scorer.score(pp[m],['ray'])['ray'])
                    sr.append(scorer)
                for m in METHODS:pools[m][e,f].append(pr[m]);values[m][e,f].append(vr[m])
                scorers[e,f].append(sr)
            progress(variant+' all 24x24 '+str((e,f)))
        np.savez_compressed(root/f'{variant}_field_correspondences.npz',source_indices=source_indices,**field_data)
        np.savez_compressed(root/f'{variant}_union_pools.npz',**pose_arrays(pools['union']))
        for method in METHODS:
            np.savez_compressed(root/f'{variant}_{method}_family_pools.npz',**pose_arrays(pools[method]))
            if method=='baseline':
                # Exact deployed strong V3 ray, not a freshly retuned baseline.
                shutil.copyfile(v3root/f'{variant}_ray.npz',root/f'{variant}_{method}.npz')
                progress(variant+' baseline byte-copy');continue
            p1,p2,selected,energies=select(pools[method],values[method],bcfg,legacy)
            for i,(j,k) in enumerate(zip(p1,p2)):
                A,B=pools[method][0,1][i][j],pools[method][0,2][i][k]
                T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1));T12=T2@inverse(T1)
                val=np.repeat(values[method][0,1][i][j],len(B))+np.tile(values[method][0,2][i][k],len(A))+.5*scorers[1,2][j][k].score(T12,['ray'])['ray']
                q=int(val.argmin());selected.append([T1[q],T2[q]]);energies.append(val[q])
            # Whole old hypotheses, including exact strong V3 ray, remain legal
            # candidates; no per-object identity mixing that breaks bijection.
            for pred in legacy.values():
                vv=[]
                for i,(j,k) in enumerate(pred['target_slots']):
                    G1,G2=pred['transforms'][i]
                    vv.append(scorers[0,1][i][j].score(G1[None],['ray'])['ray'][0]+scorers[0,2][i][k].score(G2[None],['ray'])['ray'][0]+.5*scorers[1,2][j][k].score((G2@inverse(G1))[None],['ray'])['ray'][0])
                if sum(vv)<sum(energies):p1,p2=pred['target_slots'].T;selected=pred['transforms'].copy();energies=vv
            np.savez_compressed(root/f'{variant}_{method}.npz',target_slots=np.c_[p1,p2],transforms=np.array(selected),triplet_counts=np.zeros(24,int),objectives=np.array(energies))
            progress(variant+' '+method+' selected')
        save(root/f'{variant}_diagnostics.json',dict(queries=32,coarse_anchors=3,learned_dense_and_continuous_same_logits=True,
             family_pool_sizes={m:[float(np.mean([len(p) for row in pools[m][key] for p in row])) for key in PAIRS] for m in METHODS},
             note='Equal query/anchor budgets for dense/surface/ncc/random; geometry control uses 3 old pose starts x6 surface projections; not equal FLOPs. Shared scorer, family-specific additions.'))
    progress('COMPLETE')


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));cfg=json.loads(Path(v4['base_config']).read_text(encoding='utf-8'));root=Path(v4['comparison_root']);root.mkdir(parents=True,exist_ok=False)
    training=json.loads((Path(v4['training_root'])/'training_seal.json').read_text(encoding='utf-8'));assert training['complete']
    for p,h in training['checkpoints'].items():assert sha(Path(v4['training_root'])/p)==h
    scenes=[r['scene'] for r in json.loads((Path(cfg['dataset'])/'dataset_index.json').read_text(encoding='utf-8')) if r['split']!='train']
    sources={p:sha(p) for p in [__file__,'configs/material_continuous_correspondence_v4.json','inspection_decisions/material_continuous_correspondence_v4.py',
        'inspection_decisions/material_correspondence_distribution_v3.py','inspection_decisions/material_correspondence_inference_v3.py','scripts/run_material_correspondence_v3.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,v4=v4,base_config=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8')),scenes=scenes,methods=METHODS,
        sources=sources,training_seal_sha=sha(Path(v4['training_root'])/'training_seal.json'),parent_prediction_seal_sha=sha(Path(cfg['comparison_root'])/'prediction_seal.json'),
        dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),paired_controls=[dict(baseline='baseline',candidate=m) for m in METHODS[1:]]+[dict(baseline='dense',candidate='surface')],
        shared_score='frozen V3 ordinary geometry+SIFT+ray, no learned rescoring',pool_design='common old pools plus each family additions; union is separate control',
        evaluation='4 already reused DEV scenes; not independent confirmation',GT_read_before_prediction_seal=False))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,name,cfg,v4,str(root)) for name in scenes]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,scenes=scenes,methods=METHODS,sources=sources,elapsed_s=time.perf_counter()-start,
        heldout_labels_read=False,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('V4 PREDICTIONS SEALED',flush=True)


if __name__=='__main__':main()
