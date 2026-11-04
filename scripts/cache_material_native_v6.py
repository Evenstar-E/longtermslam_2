"""Public-input native SP/LG correspondences, independently sealed before GT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_native_matching_frontend_v6 import NativeMatching,extract_epochs,pair_edges
from scripts.run_textured_material_comparison_v1 import save,sha


def job(name,cfg,base,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(output)/name;root.mkdir();matcher=NativeMatching(cfg);start=time.perf_counter()
    for v in base['variants']:
        frames=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[0] for e in range(3)]
        epochs=extract_epochs(frames,matcher,cfg);features={}
        for e,groups in enumerate(epochs):
            for i,views in enumerate(groups):
                for k,a in enumerate(views):
                    if a is None:continue
                    for key in ['uv','desc','quality','pixel','original_uv','xyz','node','raw_xyz','raw_node','crop','size','original_size','origin_offset']:features[f'{e}_{i}_{k}_{key}']=a[key]
                    features[f'{e}_{i}_{k}_frame']=a['frame']
        np.savez_compressed(root/f'{v}_native_features.npz',**features)
        def progress(msg):
            save(root/'progress.json',dict(message=v+' '+msg,elapsed_s=time.perf_counter()-start))
        edges,diag=pair_edges(epochs,matcher,progress);packed={}
        for pe,key in enumerate([(0,1),(0,2),(1,2)]):
            offsets=[0];x=[];y=[];a=[];b=[];w=[]
            for rec in edges[key]:
                x.extend(rec['x']);y.extend(rec['y']);a.extend(rec['source']);b.extend(rec['target']);w.extend(rec['weight']);offsets.append(len(w))
            for key,value in dict(offsets=np.array(offsets),x=np.array(x).reshape(-1,3),y=np.array(y).reshape(-1,3),source=np.array(a,np.int64),target=np.array(b,np.int64),weight=np.array(w)).items():packed[f'{pe}_{key}']=value
        np.savez_compressed(root/f'{v}_native_edges.npz',**packed)
        save(root/f'{v}_native_diagnostics.json',dict(**diag,features=[[[0 if a is None else len(a['uv']) for a in vv] for vv in gg] for gg in epochs],
            pair_edge_counts={str(key):[len(a['weight']) for a in vv] for key,vv in edges.items()},native_detector=True,arbitrary_dense_sampling=False,ROI_resampling_is_not_new_depth=True))
        print('V6 NATIVE',name,v,'calls',diag['matcher_calls'],'seconds',round(time.perf_counter()-start,1),flush=True)
    save(root/'progress.json',dict(message='COMPLETE',elapsed_s=time.perf_counter()-start))


def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));base=json.loads(Path(cfg['base_config']).read_text(encoding='utf-8'));root=Path(cfg['native_root']);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path('third_party/material_local_matching_v1/source_manifest.json').read_text(encoding='utf-8'))
    for p,r in manifest['files'].items():assert sha(Path('third_party/material_local_matching_v1')/p)==r['sha256']
    scenes=[r['scene'] for r in json.loads((Path(base['dataset'])/'dataset_index.json').read_text(encoding='utf-8')) if r['split']!='train']
    sources={p:sha(p) for p in [__file__,'configs/material_native_modes_v6.json','inspection_decisions/material_native_matching_frontend_v6.py','inspection_decisions/material_pretrained_v1.py']}
    save(root/'frozen_protocol.json',dict(config=cfg,base_config=base,scenes=scenes,sources=sources,official_manifest_sha=sha('third_party/material_local_matching_v1/source_manifest.json'),
        dataset_release_sha=sha(Path(base['dataset'])/'release_manifest.json'),GT_read=False,preprocessing='top2 observed views per anonymous slot, native SP ROI256 with64 unique measured pixels/view; two view-pairs cover both views using descriptor-affinity pairing; native same-epoch matches merge only with15mm measured 3D support; official adaptive LG'))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,name,cfg,base,str(root)) for name in scenes]):f.result()
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'correspondence_seal.json',dict(complete=True,sources=sources,heldout_labels_read=False,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}))
    print('V6 NATIVE CORRESPONDENCES SEALED',flush=True)


if __name__=='__main__':main()
