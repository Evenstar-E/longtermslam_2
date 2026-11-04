"""Additional prescore control, same sealed inputs and identity search states."""
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
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from inspection_decisions.material_mode_intersection_v6 import intersection_modes
from scripts.run_material_native_modes_v6 import NativeScore
from scripts.run_textured_material_comparison_v1 import save,sha

ROOT=Path('results/material_mode_intersection_v6')

def job(name,cfg,base,parent):
    torch.set_num_threads(1);root=ROOT/name;root.mkdir();parent=Path(parent)/name;bcfg=json.loads(Path('configs/textured_material_comparison_v1.json').read_text(encoding='utf-8'))
    for v in base['variants']:
        frames=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[0] for e in range(3)];lf=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
        ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
        with np.load(Path(cfg['native_root'])/name/f'{v}_native_edges.npz') as f:edges=unpack_edges(dict(f),cfg)
        with np.load(parent/f'{v}_material_modes.npz') as f:material=dict(f)
        records={};payload={'keys':material['keys']}
        for h,key in enumerate(material['keys']):
            track={k:material[f'{h}_{k}'] for k in ['x','node','weight']};rr=intersection_modes(track,cfg);records[tuple(key)]=rr;payload[f'{h}_count']=len(rr)
            for k,r in enumerate(rr):
                for field in ['T','initial','path','selected']:payload[f'{h}_{k}_{field}']=r[field]
        np.savez_compressed(root/f'{v}_intersection_modes.npz',**payload)
        with np.load(parent/f'{v}_joint_cache.npz') as f:cache=dict(f)
        with np.load(parent/f'{v}_search_graph.npz') as f:graph=dict(f)
        scorers={}
        def score(e,f,i,j,poses):
            key=e,f,i,j
            if key not in scorers:
                a,b=ep[e][i],ep[f][j];pe={(0,1):0,(0,2):1,(1,2):2}[e,f]
                scorers[key]=NativeScore(PairScore(a,b,old.matches(a,b,bcfg),groups[e][i],groups[f][j],{},frames[f],j,base),edges[pe][i*24+j])
            return scorers[key].score(poses)
        costs=cache['costs'][:,2].copy();poses=cache['transforms'][:,2].copy();origin=np.full(len(costs),-1,int);lookup={tuple(k):h for h,k in enumerate(cache['triples'])}
        for h,key in enumerate(cache['triples']):
            i,j,k=map(int,key);rr=records.get((i,j,k),[])
            if not rr:continue
            T=np.array([r['T'] for r in rr]);value=score(0,1,i,j,T[:,0])+score(0,2,i,k,T[:,1])+.5*score(1,2,j,k,T[:,1]@inverse(T[:,0]));best=int(value.argmin())
            if value[best]<costs[h]:costs[h]=value[best];poses[h]=T[best];origin[h]=best
        state_values=np.array([sum(costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)) for s in graph['states']]);state=int(state_values.argmin());slots=graph['states'][state]
        index=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(slots)]
        np.savez_compressed(root/f'{v}_shared_intersection.npz',target_slots=slots,transforms=poses[index],objectives=costs[index],triplet_counts=np.zeros(24,int),search_state=state)
        np.savez_compressed(root/f'{v}_intersection_cache.npz',triples=cache['triples'],costs=costs,transforms=poses,origins=origin,state_values=state_values)
        print('V6 INTERSECTION',name,v,'fits',sum(len(r) for r in records.values()),flush=True)

def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));base=json.loads(Path(cfg['base_config']).read_text(encoding='utf-8'));parent=Path(cfg['output']);start=time.monotonic()
    while True:
        try:seal=json.loads((parent/'prediction_seal.json').read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):seal={}
        if seal.get('complete'):break
        if time.monotonic()-start>5400:raise TimeoutError('Main prediction seal missing, no process killed')
        time.sleep(2)
    ROOT.mkdir(parents=True,exist_ok=False);scenes=json.loads((parent/'frozen_protocol.json').read_text(encoding='utf-8'))['scenes']
    sources={p:sha(p) for p in [__file__,'inspection_decisions/material_mode_intersection_v6.py','inspection_decisions/material_correspondence_mode_graph_v6.py','scripts/run_material_native_modes_v6.py']}
    save(ROOT/'frozen_protocol.json',dict(sources=sources,parent_seal_sha=sha(parent/'prediction_seal.json'),prescore=True,reason='product of correlated distance similarities suppresses all shared proposals on public-input diagnostics; same-threshold intersection, original preserved',common_identity_graph=True))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,name,cfg,base,str(parent)) for name in scenes]):f.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(ROOT/'prediction_seal.json',dict(complete=True,heldout_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.npz')}))
    print('V6 INTERSECTION SEALED',flush=True)

if __name__=='__main__':main()
