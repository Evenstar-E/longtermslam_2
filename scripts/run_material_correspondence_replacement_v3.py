"""Re-score SAME sealed union after removing duplicated legacy texture term."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,observed_groups
from inspection_decisions.material_correspondence_inference_v3 import inverse,neural_distribution,refine_joint
from inspection_decisions.material_correspondence_replacement_v3 import ReplacementScore
from scripts.run_material_correspondence_v3 import assignments
from scripts.run_textured_material_comparison_v1 import save,sha


METHODS=['ordinary','ray','hard2','soft2','soft3']


def job(name,cfg,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(output)/name;root.mkdir();source=Path(cfg['comparison_root'])/name
    bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'));models={};start=time.perf_counter()
    for mode in ['hard2','soft2','soft3']:
        ck=torch.load(Path(cfg['training_root'])/f'{mode}_31.pt',map_location='cpu',weights_only=True);model=CorrespondenceModel();model.load_state_dict(ck['state_dict']);models[mode]=model.eval()
    for v in cfg['variants']:
        frames=[load_scene_epoch(cfg['dataset'],name,e,v,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        lf=[load_scene_epoch(cfg['dataset'],name,e,v,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        le=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];gg=[observed_groups(f,cfg) for f in frames];encoded={}
        for mode,model in models.items():
            with torch.inference_mode():encoded[mode]=[[dict(xyz=g['xyz'],sigma=g['sigma'],area=g['area'],desc=model.encode(torch.from_numpy(g['patch'])).numpy()) for g in ep] for ep in gg]
        with np.load(source/f'{v}_union_pools.npz') as f:packed=dict(f)
        shutil.copyfile(source/f'{v}_union_pools.npz',root/f'{v}_union_pools.npz')
        legacy={}
        for m in ['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
            with np.load(Path('results/textured_material_comparison_v1')/name/f'{v}_{m}.npz') as f:legacy[m]=dict(f)
        pool={};scores={};values={};cost={m:{} for m in METHODS}
        for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
            pool[e,f]=[];scores[e,f]=[];values[e,f]=[]
            for m in METHODS:cost[m][e,f]=np.zeros((24,24))
            for i in range(24):
                pp=[];ss=[];vv=[]
                for j in range(24):
                    neural={mode:{**neural_distribution(model,encoded[mode][e][i],encoded[mode][f][j],cfg,mode),'a':encoded[mode][e][i],'b':encoded[mode][f][j]} for mode,model in models.items()}
                    scorer=ReplacementScore(le[e][i],le[f][j],old.matches(le[e][i],le[f][j],bcfg),encoded['soft3'][e][i],encoded['soft3'][f][j],neural,frames[f],j,cfg)
                    p=packed['transforms'][pe,i,j,:packed['counts'][pe,i,j]];val=scorer.score(p,METHODS)
                    pp.append(p);ss.append(scorer);vv.append(val)
                    for m in METHODS:cost[m][e,f][i,j]=val[m].min()
                pool[e,f].append(pp);scores[e,f].append(ss);values[e,f].append(vv)
        for m in METHODS:
            if m=='ordinary':
                for suffix in ['','_pooled']:shutil.copyfile(source/f'{v}_{m}{suffix}.npz',root/f'{v}_{m}{suffix}.npz')
                shutil.copyfile(source/f'{v}_{m}_optimization.json',root/f'{v}_{m}_optimization.json');continue
            p1,p2=assignments(cost[m],bcfg,legacy);selected=[];ee=[]
            for i,(j,k) in enumerate(zip(p1,p2)):
                A,B=pool[0,1][i][j],pool[0,2][i][k];T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1))
                val=np.repeat(values[0,1][i][j][m],len(B))+np.tile(values[0,2][i][k][m],len(A))+.5*scores[1,2][j][k].score(T2@inverse(T1),[m])[m]
                ix=int(val.argmin());selected.append([T1[ix],T2[ix]]);ee.append(val[ix])
            baseline=legacy['fusion'];be=[]
            for i,(j,k) in enumerate(baseline['target_slots']):
                A,B=baseline['transforms'][i];be.append(scores[0,1][i][j].score(A[None],[m])[m][0]+scores[0,2][i][k].score(B[None],[m])[m][0]+.5*scores[1,2][j][k].score((B@inverse(A))[None],[m])[m][0])
            used=sum(be)<sum(ee)
            if used:p1,p2=baseline['target_slots'].T;selected=baseline['transforms'].copy();ee=be
            pred=dict(target_slots=np.c_[p1,p2],transforms=np.array(selected),triplet_counts=np.zeros(24,int),objectives=np.array(ee),used_legacy_hypothesis=used)
            np.savez_compressed(root/f'{v}_{m}_pooled.npz',**pred);refined=[];paths=[]
            for i,(j,k) in enumerate(zip(p1,p2)):
                T,path=refine_joint(*pred['transforms'][i],[scores[0,1][i][j],scores[0,2][i][k],scores[1,2][j][k]],m,cfg);refined.append(T);paths.append(path.tolist())
            np.savez_compressed(root/f'{v}_{m}.npz',**{**pred,'transforms':np.array(refined),'objectives':np.array([p[-1] for p in paths])})
            save(root/f'{v}_{m}_optimization.json',dict(paths=paths,all_monotone=True))
        print('REPLACEMENT',name,v,round(time.perf_counter()-start,1),flush=True)


def main():
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'));source=Path(cfg['comparison_root']);assert (source/'prediction_seal.json').exists()
    root=Path('results/material_correspondence_replacement_v3');root.mkdir(parents=True,exist_ok=False)
    proto=json.loads((source/'frozen_protocol.json').read_text(encoding='utf-8'));sources={p:sha(p) for p in [__file__,'inspection_decisions/material_correspondence_replacement_v3.py','inspection_decisions/material_correspondence_inference_v3.py','inspection_decisions/material_correspondence_distribution_v3.py']}
    proto.update(methods=METHODS,sources=sources,method_features={m:'union' for m in METHODS},visibility_methods=[m for m in METHODS if m!='ordinary'],
        paired_controls=[dict(baseline='ray',candidate=m) for m in ['hard2','soft2','soft3']],
        stage='POSTHOC DEV structural ablation: remove old texture score, retain identical pool and model; no hyperparameter search',base_prediction_seal_sha=sha(source/'prediction_seal.json'))
    save(root/'frozen_protocol.json',proto);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,s,cfg,str(root)) for s in proto['scenes']]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,scenes=proto['scenes'],methods=METHODS,sources=sources,elapsed_s=time.perf_counter()-start,
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('REPLACEMENT PREDICTIONS SEALED',flush=True)


if __name__=='__main__':main()
