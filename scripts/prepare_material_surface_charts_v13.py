"""Public-only actual-pixel metric charts, sealed before task evaluation."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration as previous_configuration,setup
from inspection_decisions.material_surface_correspondence_v13 import build_charts
from scripts.run_textured_material_comparison_v1 import save,sha


def configuration():
    cfg=previous_configuration();cfg.update(json.loads(Path('configs/material_surface_correspondence_v13.json').read_text(encoding='utf-8')));return cfg


def main():
    cfg=configuration();parent=Path(cfg['previous_root']);index=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in index[group].items():assert sha(p)==h,p
    root=Path(cfg['chart_cache']);assert not root.exists();root.mkdir(parents=True);start=time.perf_counter();stats=[]
    sources={p:sha(p) for p in [__file__,'configs/material_surface_correspondence_v13.json','inspection_decisions/material_surface_correspondence_v13.py']}
    for v in cfg['variants']:
        _,frames,_,_,_,_,_,banks,_=setup(cfg,v);charts=[[build_charts(a,frames[e],slot,cfg) for slot,a in enumerate(row)] for e,row in enumerate(banks)]
        np.savez_compressed(root/f'{v}_charts.npz',**{f'{e}_{i}_{k}':val for e,row in enumerate(charts) for i,a in enumerate(row) for k,val in a.items()})
        stats.append(dict(variant=v,nodes=sum(len(a['xyz']) for row in charts for a in row),queried_ring_points=sum(a['mask'].size for row in charts for a in row),supported_ring_points=sum(int(a['mask'].sum()) for row in charts for a in row),textureless_nodes=sum(int((a['texture_energy']<1e-6).sum()) for row in charts for a in row)));print('V13 CHARTS',v,stats[-1],flush=True)
    save(root/'chart_seal.json',dict(complete=True,private_labels_read=False,sources=sources,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')},parent_completion_sha=sha(parent/'completion_integrity.json'),seconds=time.perf_counter()-start,statistics=stats,scope=cfg['contract']))


if __name__=='__main__':main()
