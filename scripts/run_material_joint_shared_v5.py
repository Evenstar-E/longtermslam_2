"""Shared finite ID-search graph, exact full pose pairs, frozen material fields."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher,surface_grids
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score,common_search
from inspection_decisions.material_shared_correspondence_factor_v5 import measured_bank,pixel_footprint,MaterialView,material_energies
from scripts.run_textured_material_comparison_v1 import save,sha


METHODS=['ordinary_frozen_ids','ordinary_joint','fusion','independent','shared']
PARENTS=['baseline','ncc','random','dense','surface','geo','union']
PAIRS=[(0,1),(0,2),(1,2)]


def job(name,cfg,v4,base,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(output)/name;root.mkdir();start=time.perf_counter()
    parent=Path(v4['comparison_root'])/name;bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    model=SurfaceMatcher();checkpoint=Path(v4['training_root'])/f"matcher_{cfg['feature_seed']}.pt";model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['state_dict']);model.eval()
    def progress(message,**data):
        save(root/'progress.json',dict(message=message,elapsed_s=time.perf_counter()-start,**data));print('V5',name,message,round(time.perf_counter()-start,1),flush=True)
    for variant in base['variants']:
        frames=[load_scene_epoch(base['dataset'],name,e,variant,shuffle_seed=base['seed']+e)[0] for e in range(3)]
        lf=[load_scene_epoch(base['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
        grids=[surface_grids(f,g,base,v4) for f,g in zip(frames,groups)];banks=[];queries=[];source_desc=[]
        with torch.inference_mode():
            for e in range(3):
                bb=[]
                for i,g in enumerate(groups[e]):
                    feat=model.encode(torch.from_numpy(g['patch']));array=feat.flatten(2).transpose(1,2).numpy()
                    bb.append(measured_bank(g,grids[e][i],array,cfg))
                    if e==0:
                        q=pixel_footprint(g,frames[0],cfg);queries.append(q);source_desc.append(model.query(feat).numpy()[q['indices']])
                banks.append(bb)
        with np.load(parent/f'{variant}_union_pools.npz') as f:packed=dict(f)
        shutil.copyfile(parent/f'{variant}_union_pools.npz',root/f'{variant}_union_pools.npz')
        poses={};pair_scores={};pair_values={};view_cache={};triple_cache={};calls=0
        for pe,key in enumerate(PAIRS):
            poses[key]=[[packed['transforms'][pe,i,j,:packed['counts'][pe,i,j]] for j in range(24)] for i in range(24)]
        def scorer(e,f,i,j):
            key=e,f,i,j
            if key not in pair_scores:
                a,b=ep[e][i],ep[f][j];pair_scores[key]=PairScore(a,b,old.matches(a,b,bcfg),groups[e][i],groups[f][j],{},frames[f],j,base)
            return pair_scores[key]
        def edge(e,f,i,j):
            key=e,f,i,j
            if key not in pair_values:pair_values[key]=ordinary_score(scorer(e,f,i,j),poses[e,f][i][j])
            return pair_values[key]
        def likelihood(f,i,j):
            key=f,i,j
            if key not in view_cache:
                view=MaterialView(queries[i],source_desc[i],banks[f][j],frames[f],j,cfg)
                view_cache[key]=view.likelihood(poses[0,f][i][j])
            return view_cache[key]
        def objective(i,j,k):
            nonlocal calls
            key=i,j,k
            if key not in triple_cache:
                A,B=poses[0,1][i][j],poses[0,2][i][k]
                T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1))
                ordinary=edge(0,1,i,j)[:,None]+edge(0,2,i,k)[None,:]+.5*ordinary_score(scorer(1,2,j,k),T2@inverse(T1)).reshape(len(A),len(B))
                aa,bb=likelihood(1,i,j),likelihood(2,i,k);point,ind,shared=material_energies(aa,bb,queries[i]['node_weights'],queries[i]['area'])
                values=[ordinary,ordinary+cfg['material_weight']*point,ordinary+cfg['material_weight']*ind,ordinary+cfg['material_weight']*shared]
                best=np.array([v.argmin() for v in values]);minimum=np.array([v.ravel()[p] for v,p in zip(values,best)])
                transform=np.array([[T1[p],T2[p]] for p in best]);triple_cache[key]=minimum,transform
                calls+=1
                if calls%200==0:progress(variant+' joint triples',count=calls)
            return triple_cache[key]
        pair_cost=[]
        for e,f in PAIRS:pair_cost.append(np.array([[edge(e,f,i,j).min() for j in range(24)] for i in range(24)]))
        initials=[]
        for m in PARENTS:
            with np.load(parent/f'{variant}_{m}.npz') as f:initials.append(f['target_slots'])
        with np.load(Path('results/material_continuous_retention_control_v4')/name/f'{variant}_retained.npz') as f:initials.append(f['target_slots'])
        old_allowed=[[set(int(s[i,e]) for s in initials) for i in range(24)] for e in [0,1]]
        allowed=[[old_allowed[e][i]|set(np.argsort(pair_cost[e][i])[:cfg['identity_neighbors']].tolist()) for i in range(24)] for e in [0,1]]
        search=common_search(initials,objective,allowed,pair_cost,old_allowed,rounds=cfg['search_rounds'],per_round=cfg['states_per_round'])
        np.savez_compressed(root/f'{variant}_search_graph.npz',states=search['states'],costs=search['costs'],initial_count=search['initial_count'],best=search['best'],paths=np.array(search['paths']))
        keys=np.array(list(triple_cache));np.savez_compressed(root/f'{variant}_joint_cache.npz',triples=keys,costs=np.array([triple_cache[tuple(k)][0] for k in keys]),transforms=np.array([triple_cache[tuple(k)][1] for k in keys]))
        emitted={}
        for m in METHODS:
            col=0 if m.startswith('ordinary') else ['ordinary','fusion','independent','shared'].index(m)
            state=int(search['initial_best'][0] if m=='ordinary_frozen_ids' else search['best'][col]);slots=search['states'][state]
            transforms=np.array([objective(i,int(j),int(k))[1][col] for i,(j,k) in enumerate(slots)]);energy=np.array([objective(i,int(j),int(k))[0][col] for i,(j,k) in enumerate(slots)])
            np.savez_compressed(root/f'{variant}_{m}.npz',target_slots=slots,transforms=transforms,objectives=energy,triplet_counts=np.zeros(24,int),search_state=state)
            emitted[m]=dict(state=state,total=float(energy.sum()))
        # Store selected likelihoods for independent reconstruction without GT.
        evidence={}
        for m,col in [('fusion',1),('independent',2),('shared',3)]:
            slots=search['states'][emitted[m]['state']]
            for i,(j,k) in enumerate(slots):
                T=objective(i,int(j),int(k))[1][col];A,B=poses[0,1][i][j],poses[0,2][i][k]
                p=int(np.argmin(np.max(abs(A-T[0]),axis=(1,2))));q=int(np.argmin(np.max(abs(B-T[1]),axis=(1,2))))
                evidence[f'{m}_{i}_a']=likelihood(1,i,int(j))[p];evidence[f'{m}_{i}_b']=likelihood(2,i,int(k))[q]
                evidence[f'{m}_{i}_alpha']=queries[i]['area']
        np.savez_compressed(root/f'{variant}_selected_material.npz',**evidence)
        save(root/f'{variant}_diagnostics.json',dict(common_states=len(search['states']),initials=search['initial_count'],triple_count=len(triple_cache),
            all_pose_pairs_enumerated=True,all_old_candidates_retained=True,search='two common neighborhood rounds; top3 independent-edge neighbors plus all old predicted links; finite graph not global optimum',
            emitted=emitted,bank_sizes=[[len(b['xyz']) for b in bb] for bb in banks],footprint_scale='actual +/- pixel std and +/- depth bound/sqrt3; 8 nodes, no widening'))
        progress(variant+' COMPLETE',triples=len(triple_cache),states=len(search['states']))
    progress('COMPLETE')


def main():
    cfg=json.loads(Path('configs/material_joint_shared_v5.json').read_text(encoding='utf-8'));v4=json.loads(Path(cfg['v4_config']).read_text(encoding='utf-8'));base=json.loads(Path(v4['base_config']).read_text(encoding='utf-8'))
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);parent=Path(v4['comparison_root']);p=json.loads((parent/'prediction_seal.json').read_text(encoding='utf-8'));assert p['complete']
    sources={p:sha(p) for p in [__file__,'configs/material_joint_shared_v5.json','inspection_decisions/material_joint_id_pose_search_v5.py','inspection_decisions/material_shared_correspondence_factor_v5.py',
        'inspection_decisions/material_correspondence_inference_v3.py','inspection_decisions/material_continuous_correspondence_v4.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,v4=v4,base_config=base,legacy_config=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8')),
        methods=METHODS,scenes=p['scenes'],sources=sources,shared_input='V4 measured RGB-D fields + exact identical V4 union pose pools',
        parent_prediction_seal_sha=sha(parent/'prediction_seal.json'),feature_checkpoint=str(Path(v4['training_root'])/'matcher_31.pt'),feature_checkpoint_sha=sha(Path(v4['training_root'])/'matcher_31.pt'),
        dataset_release_sha=sha(Path(base['dataset'])/'release_manifest.json'),evaluation='4 reused DEV scenes, no confirmation',
        controls=[dict(baseline='ordinary_frozen_ids',candidate='ordinary_joint'),dict(baseline='ordinary_joint',candidate='fusion'),dict(baseline='fusion',candidate='independent'),dict(baseline='independent',candidate='shared')]))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for future in as_completed([pool.submit(job,name,cfg,v4,base,str(root)) for name in p['scenes']]):future.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'prediction_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('V5 PREDICTIONS SEALED',flush=True)


if __name__=='__main__':main()
