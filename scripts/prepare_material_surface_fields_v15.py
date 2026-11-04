"""Public only: dense learned fields on original views, no new training."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel
from inspection_decisions.material_surface_distribution_matching_v15 import encode_record,add_area
from scripts.run_material_continuous_surface_correspondence_v14c import configuration as parent_configuration,setup,dense_records
from scripts.run_textured_material_comparison_v1 import save,sha


def configuration():
    c=parent_configuration();c.update(json.loads(Path('configs/material_surface_distribution_matching_v15.json').read_text(encoding='utf-8')));return c


def arrays(cfg,v):
    _,frames,_,_,_,_,views,banks,_=setup(cfg,v);dense=dense_records(views);return frames,views,banks,dense


def model(cfg):
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True);net=CorrespondenceModel();net.load_state_dict(checkpoint['state_dict']);return net.eval()


def public_normalizer(a,targets):
    z=np.zeros(len(a['xyz']))
    for b in targets:
        for start in range(0,len(a['xyz']),64):
            d=a['desc'][start:start+64]@b['desc'].T;k=np.exp((d-1)/.12);z[start:start+len(d)]+=k@b['area']/len(targets)
    return np.maximum(z,1e-12)


def job(cfg,v,profile=False):
    torch.set_num_threads(1);root=Path(cfg['field_cache']);start=time.perf_counter();frames,views,banks,dense=arrays(cfg,v);net=model(cfg);fc=json.loads(Path(cfg['feature_config']).read_text(encoding='utf-8'));payload={};source_count=0;target_count=0
    for e in range(3):
        slots=[int(np.argmax([len(a['xyz']) for a in dense[e]]))] if profile else range(24)
        for i in slots:
            a,b=banks[e][i],dense[e][i];a['desc']=encode_record(net,a,frames[e],fc,cfg).astype(np.float16).astype(np.float32);b['desc']=encode_record(net,b,views[e][i],fc,cfg).astype(np.float16).astype(np.float32)
            for record in [a,b]:record['desc']/=np.maximum(np.linalg.norm(record['desc'],axis=1,keepdims=True),1e-8)
            add_area(b,views[e][i],i);payload[f'a_{e}_{i}_desc']=a['desc'].astype(np.float16);payload[f'd_{e}_{i}_desc']=b['desc'].astype(np.float16);payload[f'd_{e}_{i}_area']=b['area'];source_count+=len(a['xyz']);target_count+=len(b['xyz'])
            if profile:return dict(variant=v,epoch=e,slot=i,native_pixels=len(b['xyz']),source_centers=len(a['xyz']),seconds=time.perf_counter()-start,private_labels_read=False)
            if i%4==3:save(root/f'{v}_progress.json',dict(epoch=e,slot=i,seconds=time.perf_counter()-start,source_centers=source_count,native_pixels=target_count))
    # Reproduce exactly what subsequent loaders see after float16 storage.
    for e in range(3):
        for i in range(24):
            for key,record in [('a',banks[e][i]),('d',dense[e][i])]:
                record['desc']=payload[f'{key}_{e}_{i}_desc'].astype(np.float32);record['desc']/=np.maximum(np.linalg.norm(record['desc'],axis=1,keepdims=True),1e-8)
    for e in range(3):
        for f in range(3):
            if e==f:continue
            for i,a in enumerate(banks[e]):payload[f'a_{e}_{i}_normalizer_{f}']=public_normalizer(a,dense[f])
    np.savez_compressed(root/f'{v}_fields.npz',**payload);save(root/f'{v}_diagnostics.json',dict(complete=True,private_labels_read=False,source_centers=source_count,native_pixels=target_count,seconds=time.perf_counter()-start));print('V15 FIELDS',v,round(time.perf_counter()-start,1),flush=True)


def load_fields(cfg,v,banks,dense):
    with np.load(Path(cfg['field_cache'])/f'{v}_fields.npz') as z:
        for e in range(3):
            for i in range(24):
                for key,record in [('a',banks[e][i]),('d',dense[e][i])]:
                    record['desc']=z[f'{key}_{e}_{i}_desc'].astype(np.float32);record['desc']/=np.maximum(np.linalg.norm(record['desc'],axis=1,keepdims=True),1e-8)
                dense[e][i]['area']=z[f'd_{e}_{i}_area']
                for f in range(3):
                    if e!=f:banks[e][i][f'normalizer_{f}']=z[f'a_{e}_{i}_normalizer_{f}']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');args=ap.parse_args();c=configuration()
    if args.profile:print(json.dumps(job(c,'rich',True),indent=2));return
    parent=Path(c['parent_root']);completion=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in completion[group].items():assert sha(p)==h,p
    seal=json.loads(Path('results/material_correspondence_training_v3/training_seal.json').read_text(encoding='utf-8'));assert seal['train_only'] and sha(c['checkpoint'])==seal['checkpoints'][Path(c['checkpoint']).name]
    root=Path(c['field_cache']);root.mkdir(parents=True,exist_ok=False);paths=[__file__,'configs/material_surface_distribution_matching_v15.json','inspection_decisions/material_surface_distribution_matching_v15.py','inspection_decisions/material_correspondence_distribution_v3.py','inspection_decisions/material_joint_descriptor_v1.py'];sources={p:sha(p) for p in paths};save(root/'frozen_protocol.json',dict(config=c,sources=sources,parent_completion_sha=sha(parent/'completion_integrity.json'),checkpoint_sha=sha(c['checkpoint']),private_labels_read=False,new_training=False));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        for future in as_completed([pool.submit(job,c,v) for v in c['variants']]):future.result()
    for p,h in sources.items():assert sha(p)==h,p
    save(root/'field_seal.json',dict(complete=True,private_labels_read=False,new_training=False,sources=sources,checkpoint_sha=sha(c['checkpoint']),files={p.name:sha(p) for p in root.glob('*_fields.npz')},seconds=time.perf_counter()-start));print('V15 FIELD SEALED',flush=True)


if __name__=='__main__':main()
