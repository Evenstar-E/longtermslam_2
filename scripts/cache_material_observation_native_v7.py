"""All24 native correspondence cache under paired acquisition conditions."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.material_observation_inputs_v7 import load_epochs
from inspection_decisions.material_native_matching_frontend_v6 import NativeMatching,extract_epochs,pair_edges
from scripts.run_textured_material_comparison_v1 import save,sha


def job(cfg,condition,variant,native_cfg):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();root=Path(cfg['native_root'])/condition;frames,ids=load_epochs(cfg,condition,variant);matcher=NativeMatching(native_cfg);epochs=extract_epochs(frames,matcher,native_cfg);features={}
    for e,groups in enumerate(epochs):
        features[f'{e}_frame_ids']=np.array(ids[e])
        for i,views in enumerate(groups):
            for k,a in enumerate(views):
                if a is None:continue
                for key in ['uv','desc','quality','pixel','original_uv','xyz','node','raw_xyz','raw_node','crop','size','original_size','origin_offset','frame']:features[f'{e}_{i}_{k}_{key}']=a[key]
    np.savez_compressed(root/f'{variant}_features.npz',**features)
    def progress(msg):save(root/f'{variant}_progress.json',dict(message=msg,elapsed_s=time.perf_counter()-start))
    edges,diag=pair_edges(epochs,matcher,progress);packed={}
    for pe,key in enumerate([(0,1),(0,2),(1,2)]):
        offsets=[0]
        for r in edges[key]:offsets.append(offsets[-1]+len(r['weight']))
        packed[f'{pe}_offsets']=np.array(offsets)
        for field in ['x','y','source','target','weight']:packed[f'{pe}_{field}']=np.concatenate([r[field] for r in edges[key]],axis=0)
    np.savez_compressed(root/f'{variant}_edges.npz',**packed);save(root/f'{variant}_diagnostics.json',dict(**diag,features=[[[len(a['uv']) if a is not None else 0 for a in views] for views in groups] for groups in epochs],elapsed_s=time.perf_counter()-start))
    print('V7 NATIVE',condition,variant,round(time.perf_counter()-start,1),flush=True)


def main():
    path=Path('configs/material_observation_probe_v7.json');cfg=json.loads(path.read_text(encoding='utf-8'));data=Path(cfg['dataset']);start=time.monotonic()
    while not (data/'release_manifest.json').exists():
        if time.monotonic()-start>3600:raise TimeoutError('New TRAIN release missing')
        time.sleep(2)
    native_cfg=json.loads(Path(cfg['native_config']).read_text(encoding='utf-8'));root=Path(cfg['native_root']);root.mkdir(parents=True,exist_ok=False)
    for c in cfg['conditions']:(root/c).mkdir()
    official=Path('third_party/material_local_matching_v1');manifest=json.loads((official/'source_manifest.json').read_text(encoding='utf-8'))
    for p,r in manifest['files'].items():assert sha(official/p)==r['sha256']
    files=[str(path),__file__,'inspection_decisions/material_observation_inputs_v7.py','inspection_decisions/material_native_matching_frontend_v6.py','inspection_decisions/material_pretrained_v1.py',cfg['native_config']];sources={p:sha(p) for p in files}
    save(root/'frozen_protocol.json',dict(config=cfg,native_config=native_cfg,sources=sources,dataset_release_sha=sha(data/'release_manifest.json'),official_manifest_sha=sha(official/'source_manifest.json'),private_labels_read=False,acquisition_groups_input=False))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,c,v,native_cfg) for c in cfg['conditions'] for v in cfg['variants']]):f.result()
    assert all(sha(p)==h for p,h in sources.items());save(root/'correspondence_seal.json',dict(complete=True,private_labels_read=False,sources=sources,elapsed_s=time.perf_counter()-start,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*.npz')}));print('V7 NATIVE SEALED',flush=True)


if __name__=='__main__':main()
