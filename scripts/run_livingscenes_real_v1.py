"""Published shape-prior association and ordinary same-descriptor controls, sealed before GT."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from inspection_decisions.livingscenes_cpu_runtime_v1 import load_solver

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/livingscenes_real_v1');args=parser.parse_args()
    cfgpath=Path('configs/livingscenes_real_v1.json');cfg=read(cfgpath);out=Path(args.output);out.mkdir(exist_ok=False)
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    root=Path(cfg['public_root']);public=read(root/'public.json')
    save(out/'config.json',cfg)
    sources=[Path(__file__),Path('inspection_decisions/livingscenes_cpu_runtime_v1.py'),Path('inspection_decisions/pytorch3d_cpu_knn_v1.py')]
    port=Path('third_party/LivingScenes_cpu_port_v1')
    save(out/'start_manifest.json',{'sources_sha256':{str(p):sha(p) for p in sources},'config_sha256':sha(cfgpath),
        'port_sha256':{str(p.relative_to(port)):sha(p) for p in port.rglob('*.py')},
        'public_input_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'public').rglob('*.npz')},
        'persistent_ID_access':'No evaluation files read by runner or model'})
    solver=load_solver(Path.cwd(),cfg)
    import torch
    start=time.perf_counter();encodes=[];predictions=[]
    from lib_more.matcher_new import sequential_matcher
    for scene in cfg['spaces']:
        encoded={};items={};centroids={}
        for capture in ['0','1']:
            items[capture]=[r for r in public[scene][capture] if r['eligible']]
            codes=[];centroids[capture]=[]
            for entry in items[capture]:
                with np.load(root/entry['path']) as z:xyz=z['xyz'].astype(np.float32)
                tensor=torch.from_numpy(xyz).T[None];mask=torch.ones((1,1,len(xyz)),dtype=torch.bool)
                began=time.perf_counter()
                with torch.no_grad():code=solver.model.encode_fps(tensor,mask)
                if not all(torch.isfinite(v).all() for v in code.values()):raise ValueError('Nonfinite model output')
                target=out/'embeddings'/scene/capture;target.mkdir(exist_ok=True,parents=True)
                np.savez_compressed(target/(entry['token']+'.npz'),**{k:v.numpy() for k,v in code.items()})
                codes.append(code);centroids[capture].append(xyz.mean(0))
                row=dict(scene=scene,capture=capture,token=entry['token'],raw_points=len(xyz),zero_padded_fps_points=max(0,1024-len(xyz)),elapsed_s=time.perf_counter()-began)
                encodes.append(row);save(out/'monitor.json',{'encoded':len(encodes),'elapsed_s':time.perf_counter()-start,'latest':row})
                print('ENCODE',scene,capture,entry['token'],len(encodes),round(row['elapsed_s'],2),flush=True)
            encoded[capture]={k:torch.cat([v[k] for v in codes]) for k in codes[0]}
        H=items['0'];C=items['1'];hh=encoded['0']['z_inv'];cc=encoded['1']['z_inv']
        hnorm=torch.nn.functional.normalize(hh,dim=1);cnorm=torch.nn.functional.normalize(cc,dim=1)
        similarity=(hnorm@cnorm.T).numpy();np.save(out/(scene+'_similarity.npy'),similarity)
        mappings={m:{} for m in cfg['methods']}
        match=sequential_matcher(hh,cc)
        mappings['more_sequential']={int(c):int(h) for c,h in enumerate(match['matches1']) if h>=0}
        for cls in sorted({c['semantic_class'] for c in C}):
            hi=[i for i,h in enumerate(H) if h['semantic_class']==cls];ci=[i for i,c in enumerate(C) if c['semantic_class']==cls]
            if not hi:continue
            matched=sequential_matcher(hh[hi],cc[ci])
            mappings['more_classwise_sequential'].update({ci[k]:hi[int(h)] for k,h in enumerate(matched['matches1']) if h>=0})
            costs=(1-similarity[np.ix_(hi,ci)].T)/2
            ii,jj=linear_sum_assignment(costs)
            mappings['more_classwise_hungarian'].update({ci[int(i)]:hi[int(j)] for i,j in zip(ii,jj)})
            a=np.array(centroids['1'])[ci];b=np.array(centroids['0'])[hi]
            position=np.minimum(np.linalg.norm(a[:,None]-b[None],axis=-1),1.)
            fused=(costs+position)/2
            augmented=np.concatenate([fused,np.full((len(ci),len(ci)),cfg['position_control_null_cost'])],axis=1)
            ii,jj=linear_sum_assignment(augmented)
            mappings['more_position_hungarian'].update({ci[int(i)]:hi[int(j)] for i,j in zip(ii,jj) if j<len(hi)})
        for method,mapping in mappings.items():
            for i,c in enumerate(C):
                j=mapping.get(i)
                predictions.append(dict(scene=scene,method=method,current=c['token'],history=None if j is None else H[j]['token'],
                    cosine_similarity=None if j is None else float(similarity[j,i])))
    save(out/'encoding_stats.json',encodes);save(out/'predictions.json',predictions)
    save(out/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,'encoded_objects':len(encodes),
        'sha256':{p.name:sha(p) for p in [out/'config.json',out/'predictions.json',out/'encoding_stats.json',out/'start_manifest.json']},'uses_current_GT_masks':True,'uses_persistent_ID_answers':False})
    print('SEALED',len(predictions),len(encodes),flush=True)

if __name__=='__main__':main()
