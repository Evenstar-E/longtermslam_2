"""Measured pixel/patch cache and official Open3D FPFH/FGR geometry proposals."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from inspection_decisions.material_observation_inputs_v7 import load_epochs
from inspection_decisions.material_surface_patch_registration_v9 import sample_surfaces,geometry_features,global_pose,icp_fit
from scripts.run_textured_material_comparison_v1 import save,sha

PAIRS=[(0,1),(0,2),(1,2)]

def unpack(path):
    with np.load(path) as f:return [[{k[len(f'{e}_{i}_'):]:f[k] for k in f if k.startswith(f'{e}_{i}_')} for i in range(24)] for e in range(3)]

def geom_job(cfg,pe):
    root=Path(cfg['geometry_cache']);banks=unpack(root/'flat_surfaces.npz');e,f=PAIRS[pe];out=np.tile(np.eye(4),(24,24,2,1,1));start=time.perf_counter()
    for i in range(24):
        for j in range(24):
            T=global_pose(banks[e][i],banks[f][j],cfg,cfg['seed']+pe*576+i*24+j);out[i,j,0]=T;out[i,j,1]=icp_fit(banks[e][i],banks[f][j],T,cfg)[0]
        if i%4==3:save(root/f'pair{pe}_progress.json',dict(source_slot=i,elapsed_s=time.perf_counter()-start))
    np.savez_compressed(root/f'pair{pe}_geometry.npz',transforms=out);print('SURFACE GEOMETRY',pe,round(time.perf_counter()-start,1),flush=True)

def main():
    cp=Path('configs/material_surface_patch_registration_v9.json');cfg=json.loads(cp.read_text(encoding='utf-8'));ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));parent=Path(cfg['parent_root']);seal=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in seal[group].items():assert sha(p)==h,p
    root=Path(cfg['geometry_cache']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter();geometry=None;statistics=[]
    for v in cfg['variants']:
        frames,ids=load_epochs(ic,cfg['condition'],v);banks=[sample_surfaces(f,cfg) for f in frames]
        for e in range(3):
            for i,r in enumerate(banks[e]):
                r['camera']=np.array([frames[e][int(fi)]['C'][:3,3] for fi in r['frame']])
                if geometry is None:r['normal'],r['fpfh']=geometry_features(r,cfg)
                else:
                    assert np.array_equal(r['xyz'],geometry[e][i]['xyz']) and np.array_equal(r['patch_xyz'],geometry[e][i]['patch_xyz']);r['normal']=geometry[e][i]['normal'];r['fpfh']=geometry[e][i]['fpfh']
        if geometry is None:geometry=banks
        np.savez_compressed(root/f'{v}_surfaces.npz',**{f'{e}_{i}_{k}':a for e in range(3) for i,r in enumerate(banks[e]) for k,a in r.items()});statistics.append(dict(variant=v,points=sum(len(r['xyz']) for row in banks for r in row),patches=sum(len(r['patch_xyz']) for row in banks for r in row)));print('SURFACES',v,statistics[-1],flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(geom_job,cfg,pe) for pe in range(3)]):f.result()
    import open3d as o3d
    sources=[__file__,'inspection_decisions/material_surface_patch_registration_v9.py',str(cp),cfg['input_config']];save(root/'surface_seal.json',dict(complete=True,private_labels_read=False,statistics=statistics,elapsed_s=time.perf_counter()-start,open3d_version=o3d.__version__,sources={p:sha(p) for p in sources},files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')},data_release_sha=sha(Path(ic['dataset'])/'release_manifest.json'),meaning='All samples are measured pixels; voxel representative and patch raw pixel provenance retained. Official Open3D FPFH/FGR+point-to-planeICP, not original algorithm.'));print('SURFACE CACHE SEALED',flush=True)

if __name__=='__main__':main()
