"""Prescore control: identical V4 search with old-only poses, no new field matches."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from scripts.run_material_continuous_v4 import select,PAIRS
from scripts.run_textured_material_comparison_v1 import save,sha


CONTROL_ROOT=Path('results/material_continuous_retention_control_v4')


def job(name,cfg,mainroot,output):
    dest=Path(output)/name;dest.mkdir();bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    for variant in cfg['variants']:
        frames=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=cfg['seed']+e)[0] for e in range(3)]
        lf=[load_scene_epoch(cfg['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,cfg) for f in frames];legacy={}
        for m in ['geometry','texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
            with np.load(Path('results/textured_material_comparison_v1')/name/f'{variant}_{m}.npz') as f:legacy[m]=dict(f)
        with np.load(Path(cfg['comparison_root'])/name/f'{variant}_ray.npz') as f:legacy['v3_ray']=dict(f)
        source=Path(mainroot)/name/f'{variant}_baseline_family_pools.npz';shutil.copyfile(source,dest/f'{variant}_retained_family_pools.npz')
        with np.load(source) as f:packed=dict(f)
        pools={};scorers={};values={}
        for pe,(e,f) in enumerate(PAIRS):
            pools[e,f]=[];scorers[e,f]=[];values[e,f]=[]
            for i in range(24):
                pr=[];sr=[];vr=[]
                for j in range(24):
                    a,b=ep[e][i],ep[f][j];mm=old.matches(a,b,bcfg);score=PairScore(a,b,mm,groups[e][i],groups[f][j],{},frames[f],j,cfg)
                    pp=packed['transforms'][pe,i,j,:packed['counts'][pe,i,j]];pr.append(pp);sr.append(score);vr.append(score.score(pp,['ray'])['ray'])
                pools[e,f].append(pr);scorers[e,f].append(sr);values[e,f].append(vr)
        p1,p2,selected,energies=select(pools,values,bcfg,legacy)
        for i,(j,k) in enumerate(zip(p1,p2)):
            A,B=pools[0,1][i][j],pools[0,2][i][k];T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1))
            value=np.repeat(values[0,1][i][j],len(B))+np.tile(values[0,2][i][k],len(A))+.5*scorers[1,2][j][k].score(T2@inverse(T1),['ray'])['ray']
            q=int(value.argmin());selected.append([T1[q],T2[q]]);energies.append(value[q])
        for pred in legacy.values():
            vv=[]
            for i,(j,k) in enumerate(pred['target_slots']):
                G1,G2=pred['transforms'][i]
                vv.append(scorers[0,1][i][j].score(G1[None],['ray'])['ray'][0]+scorers[0,2][i][k].score(G2[None],['ray'])['ray'][0]+.5*scorers[1,2][j][k].score((G2@inverse(G1))[None],['ray'])['ray'][0])
            if sum(vv)<sum(energies):p1,p2=pred['target_slots'].T;selected=pred['transforms'].copy();energies=vv
        np.savez_compressed(dest/f'{variant}_retained.npz',target_slots=np.c_[p1,p2],transforms=np.array(selected),triplet_counts=np.zeros(24,int),objectives=np.array(energies))
        print('V4 RETAINED SEARCH CONTROL',name,variant,flush=True)


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));cfg=json.loads(Path(v4['base_config']).read_text(encoding='utf-8'));mainroot=Path(v4['comparison_root'])
    parent=json.loads((mainroot/'prediction_seal.json').read_text(encoding='utf-8'));assert parent['complete'];CONTROL_ROOT.mkdir(parents=True,exist_ok=False)
    sources={p:sha(p) for p in [__file__,'scripts/run_material_continuous_v4.py','inspection_decisions/material_correspondence_inference_v3.py','inspection_decisions/material_correspondence_distribution_v3.py',
        'scripts/score_material_continuous_v4.py','research/material_continuous_correspondence_protocol_v4_2026-10-26.md']}
    save(CONTROL_ROOT/'frozen_protocol.json',dict(prescore_addendum=True,reason='Isolate identical V4 old whole-hypothesis search from new proposal contribution; no GT read',
        main_prediction_seal_sha=sha(mainroot/'prediction_seal.json'),sources=sources,methods=['retained'],scenes=parent['scenes']))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,name,cfg,str(mainroot),str(CONTROL_ROOT)) for name in parent['scenes']]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(CONTROL_ROOT/'prediction_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,
        files={str(p.relative_to(CONTROL_ROOT)):sha(p) for p in CONTROL_ROOT.rglob('*.npz')}))


if __name__=='__main__':main()
