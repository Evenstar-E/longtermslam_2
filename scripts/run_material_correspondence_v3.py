"""Full 24-asset inference with common legacy/official/graph proposal union."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,observed_groups
from inspection_decisions.material_correspondence_inference_v3 import METHODS,PairScore,inverse,unique_poses,spectral_pose,neural_distribution,refine_joint
from scripts.run_textured_material_comparison_v1 import save,sha


def assignments(cost,cfg,legacy):
    def perm(c):
        i,j=linear_sum_assignment(c);out=np.empty(24,int);out[i]=j;return out
    starts=[old.solve_assignments(cost,cfg)]+[(p['target_slots'][:,0],p['target_slots'][:,1]) for p in legacy.values()]
    best=None
    for p1,p2 in starts:
        p1,p2=p1.copy(),p2.copy()
        for _ in range(cfg['assignment_iterations']):
            p1=perm(cost[0,1]+.5*cost[1,2][:,p2].T)
            p2=perm(cost[0,2]+.5*cost[1,2][p1,:])
        value=np.sum(cost[0,1][np.arange(24),p1]+cost[0,2][np.arange(24),p2]+.5*cost[1,2][p1,p2])
        if best is None or value<best[0]:best=value,p1.copy(),p2.copy()
    return best[1],best[2]


def job(name,cfg,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(output)/name;root.mkdir();start=time.perf_counter()
    def progress(message):
        save(root/'progress.json',dict(message=message,elapsed_s=time.perf_counter()-start));print('V3',name,message,round(time.perf_counter()-start,1),flush=True)
    models={}
    for mode in ['hard2','soft2','soft3']:
        ck=torch.load(Path(cfg['training_root'])/f'{mode}_31.pt',map_location='cpu',weights_only=True)
        model=CorrespondenceModel(64,cfg['temperature']);model.load_state_dict(ck['state_dict']);models[mode]=model.eval()
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'));gp=None
    oldroot=Path('results/textured_material_comparison_v1')/name
    legacy_methods=['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']
    for variant in cfg['variants']:
        frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        legacy_frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        legacy_ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(legacy_frames)]
        if gp is None:gp=old.geometric_pools(legacy_ep,bcfg)
        groups=[observed_groups(f,cfg) for f in frames]
        encoded={mode:[] for mode in models}
        for mode,model in models.items():
            for gg in groups:
                ep=[]
                with torch.inference_mode():
                    for g in gg:
                        desc=model.encode(torch.from_numpy(g['patch'])).numpy()
                        ep.append(dict(xyz=g['xyz'],area=g['area'],sigma=g['sigma'],desc=desc))
                encoded[mode].append(ep)
        legacy={}
        for m in legacy_methods:
            with np.load(oldroot/f'{variant}_{m}.npz') as f:legacy[m]=dict(f)
        official_path=Path('results/material_joint_descriptor_comparison_v1')/name/f'{variant}_superpoint_pools.npz'
        with np.load(official_path) as f:official=dict(f)
        pools={};pair_scores={};values={};cost={m:{} for m in METHODS}
        for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
            pools[e,f]=[];pair_scores[e,f]=[];values[e,f]=[]
            for m in METHODS:cost[m][e,f]=np.zeros((24,24))
            for i in range(24):
                poolrow=[];scorerrow=[];valuerow=[]
                for j in range(24):
                    a,b=legacy_ep[e][i],legacy_ep[f][j];mm=old.matches(a,b,bcfg)
                    neural={mode:{**neural_distribution(model,encoded[mode][e][i],encoded[mode][f][j],cfg,mode),
                        'a':encoded[mode][e][i],'b':encoded[mode][f][j]} for mode,model in models.items()}
                    proposals=list(gp[e,f][i][j])
                    T=old.ransac(a,b,mm,bcfg,bcfg['seed']+e*10000+f*1000+i*24+j)
                    if T is not None:proposals.extend([T,old.icp(a,b,T,bcfg)])
                    T=spectral_pose(a,b,None,cfg,mm)
                    if T is not None:proposals.extend([T,old.icp(a,b,T,bcfg)])
                    proposals.extend(official['transforms'][pe,i,j,:official['counts'][pe,i,j]])
                    for mode in models:
                        T=spectral_pose(encoded[mode][e][i],encoded[mode][f][j],neural[mode]['q'],cfg)
                        if T is not None:proposals.extend([T,old.icp(a,b,T,bcfg)])
                    # Exact old complete hypotheses stay available, including
                    # the old current-current composed transform.
                    for pred in legacy.values():
                        slots=np.c_[np.arange(24),pred['target_slots']];GG=np.tile(np.eye(4),(24,3,1,1));GG[:,1:]=pred['transforms']
                        h=np.flatnonzero((slots[:,e]==i)&(slots[:,f]==j))
                        for hh in h:proposals.append(GG[hh,f]@inverse(GG[hh,e]))
                    pool=unique_poses(proposals)
                    scorer=PairScore(a,b,mm,encoded['soft3'][e][i],encoded['soft3'][f][j],neural,frames[f],j,cfg)
                    vv=scorer.score(pool)
                    for m in METHODS:cost[m][e,f][i,j]=vv[m].min()
                    poolrow.append(pool);scorerrow.append(scorer);valuerow.append(vv)
                pools[e,f].append(poolrow);pair_scores[e,f].append(scorerrow);values[e,f].append(valuerow)
            progress(variant+' union '+str((e,f)))
        # Pack the COMPLETE variable-length common pool, no branch top-k.
        cap=max(len(p) for mat in pools.values() for row in mat for p in row)
        packed=np.tile(np.eye(4),(3,24,24,cap,1,1));counts=np.zeros((3,24,24),int)
        for pe,key in enumerate([(0,1),(0,2),(1,2)]):
            for i in range(24):
                for j in range(24):p=pools[key][i][j];counts[pe,i,j]=len(p);packed[pe,i,j,:len(p)]=p
        np.savez_compressed(root/f'{variant}_union_pools.npz',transforms=packed,counts=counts)
        for method in METHODS:
            p1,p2=assignments(cost[method],bcfg,legacy);selected=[];energies=[]
            for i,(j,k) in enumerate(zip(p1,p2)):
                A,B=pools[0,1][i][j],pools[0,2][i][k]
                T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1));T12=T2@inverse(T1)
                v=np.repeat(values[0,1][i][j][method],len(B))+np.tile(values[0,2][i][k][method],len(A))
                v+=.5*pair_scores[1,2][j][k].score(T12,[method])[method]
                ix=int(v.argmin());selected.append([T1[ix],T2[ix]]);energies.append(v[ix])
            # Evaluate the unchanged whole old fusion hypothesis as another
            # common initialization; never mix per-asset IDs and break bijection.
            base=legacy['fusion'];oldvalues=[]
            for i,(j,k) in enumerate(base['target_slots']):
                G1,G2=base['transforms'][i];sc=[pair_scores[0,1][i][j],pair_scores[0,2][i][k],pair_scores[1,2][j][k]]
                oldvalues.append(sc[0].score(G1[None],[method])[method][0]+sc[1].score(G2[None],[method])[method][0]+.5*sc[2].score((G2@inverse(G1))[None],[method])[method][0])
            used_old=sum(oldvalues)<sum(energies)
            if used_old:p1,p2=base['target_slots'].T;selected=base['transforms'].copy();energies=oldvalues
            pred=dict(target_slots=np.c_[p1,p2],transforms=np.array(selected),triplet_counts=np.zeros(24,int),objectives=np.array(energies),used_legacy_hypothesis=used_old)
            np.savez_compressed(root/f'{variant}_{method}_pooled.npz',**pred)
            refined=[];paths=[]
            for i,(j,k) in enumerate(zip(p1,p2)):
                pair=[pair_scores[0,1][i][j],pair_scores[0,2][i][k],pair_scores[1,2][j][k]]
                TT,path=refine_joint(*pred['transforms'][i],pair,method,cfg);refined.append(TT);paths.append(path.tolist())
            np.savez_compressed(root/f'{variant}_{method}.npz',**{**pred,'transforms':np.array(refined),'objectives':np.array([x[-1] for x in paths])})
            save(root/f'{variant}_{method}_optimization.json',dict(paths=paths,all_monotone=all(np.max(np.diff(p))<=1e-10 for p in paths if len(p)>1)))
            progress(variant+' '+method+' refined')
        save(root/f'{variant}_diagnostics.json',dict(pool_cap=cap,identity_edges=3*24*24,
            counts=counts,old_hypotheses_retained=legacy_methods,
            features=[[len(g['xyz']) for g in gg] for gg in groups]))
    progress('COMPLETE')


def main():
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'))
    root=Path(cfg['comparison_root']);root.mkdir(parents=True,exist_ok=False)
    tr=Path(cfg['training_root']);s=json.loads((tr/'training_seal.json').read_text(encoding='utf-8'));assert s['complete']
    for p,h in s['checkpoints'].items():assert sha(tr/p)==h
    scenes=[r['scene'] for r in json.loads((Path(cfg['dataset'])/'dataset_index.json').read_text(encoding='utf-8')) if r['split']!='train']
    sources={p:sha(p) for p in [__file__,'configs/material_correspondence_distribution_v3.json',
        'inspection_decisions/material_correspondence_distribution_v3.py','inspection_decisions/material_correspondence_inference_v3.py',
        'inspection_decisions/textured_material_matching_v1.py']}
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    save(root/'frozen_protocol.json',dict(config={**cfg,'variants':cfg['variants']},base_config=bcfg,scenes=scenes,methods=METHODS,sources=sources,
        method_features={m:'union' for m in METHODS},visibility_methods=[m for m in METHODS if m not in ['ordinary','soft3_no_visibility']],
        paired_controls=[dict(baseline='ordinary',candidate=m) for m in METHODS if m!='ordinary'],
        dataset_release_sha=sha(Path(cfg['dataset'])/'release_manifest.json'),training_seal_sha=sha(tr/'training_seal.json'),
        legacy_prediction_seal_sha=sha('results/textured_material_comparison_v1/prediction_seal.json'),
        official_prediction_seal_sha=sha('results/material_joint_descriptor_comparison_v1/prediction_seal.json'),
        stages=['common full pool','4 common joint SE3 steps with actual backtracking'],evaluation='4 already seen DEV scenes, not independent confirmation'))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,name,cfg,str(root)) for name in scenes]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,scenes=scenes,methods=METHODS,sources=sources,
        elapsed_s=time.perf_counter()-start,heldout_labels_read=False,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('V3 PREDICTIONS SEALED',flush=True)


if __name__=='__main__':main()
