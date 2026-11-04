from pathlib import Path
import time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.visible_sampling_centroid_v1 import Model,fuse_surface


def main():
    cp=Path('configs/visible_sampling_centroid_v1.json');cfg=read(cp);root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for p,h in read(pool/'prediction_seal.json')['sha256'].items():assert sha(pool/p)==h
    fcpath=Path('results/ray_material_full_resolution_v1/effective_configuration.json');fc=read(fcpath)
    sources=[cp,Path(__file__),Path('inspection_decisions/visible_sampling_centroid_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py'),Path('inspection_decisions/real_temporal_v1.py'),Path('scripts/run_rigid_identity_bounds_v1.py')]
    inputs=[root/'public.json',fcpath]+list(pool.glob('pair_*/predictions.json'))+[p for p in (root/'public').rglob('*') if p.is_file()]
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},cfg=cfg,GT_read=False))
    frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']};began=time.perf_counter()
    for pth in sorted(pool.glob('pair_*/predictions.json')):
        p=read(pth);pair=p['pair'];history=frames[pair['history_stem']];current=frames[pair['current_stem']]
        inputs=[(history,np.eye(4))]+[(frames[b['stem']],np.array(b['T_bridge_from_history'])) for b in p['bridges']]
        assert pair['current_stem'] not in [pair['history_stem']]+[b['stem'] for b in p['bridges']]
        surface=fuse_surface(inputs,history.xyz.mean(0),cfg);model=Model(surface,current,cfg);rows=[]
        dest=out/pth.parent.name;dest.mkdir();np.savez_compressed(dest/'surface.npz',**surface)
        for k,r in enumerate(p['rows']):
            scores,diag=model.score(np.array(r['T']),r['base']);rows.append(dict(scores=scores,diagnosis=diag))
            if (k+1)%200==0:print('PROGRESS',pth.parent.name,k+1,'of',len(p['rows']),flush=True)
        choices={m:min(range(len(rows)),key=lambda k:rows[k]['scores'][m]) for m in rows[0]['scores']}
        save(dest/'predictions.json',dict(pair=pair,choices=choices,rows=rows,source_frames=[pair['history_stem']]+[b['stem'] for b in p['bridges']],surface_points=len(surface['x']),material_cells=len(surface['centers']),GT_read=False))
        print('DONE',pth.parent.name,round(time.perf_counter()-began,2),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},GT_read=False))


if __name__=='__main__':main()
