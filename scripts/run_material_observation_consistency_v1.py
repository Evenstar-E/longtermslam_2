"""Full frozen real pose pool; complete and seal before annotation scoring."""
from pathlib import Path
import time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.material_observation_consistency_v1 import pair_support


def main():
    cfgpath=Path('configs/material_observation_consistency_v1.json');cfg=read(cfgpath)
    root=Path(cfg['public_root']);poolroot=Path(cfg['pool_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for name,h in read(poolroot/'prediction_seal.json')['sha256'].items():assert sha(poolroot/name)==h
    sources=[cfgpath,Path(__file__),Path('inspection_decisions/material_observation_consistency_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py'),Path('inspection_decisions/ray_conditioned_plane_polytope_v1.py')]
    fcpath=poolroot/'effective_configuration.json'
    inputs=[root/'public.json',fcpath]+[p for p in (root/'public').rglob('*') if p.is_file()]+list(poolroot.glob('pair_*/predictions.json'))
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},config=cfg,GT_read=False))
    fc=read(fcpath);meta=read(root/'public.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in meta['clouds']}
    runs=[];start=time.perf_counter()
    for path in sorted(poolroot.glob('pair_*/predictions.json')):
        p=read(path);pair=p['pair'];h=frames[pair['history_stem']];c=frames[pair['current_stem']];rows=[]
        for entry in p['pool']:
            support=pair_support(h,c,np.array(entry['T_current_from_history']),cfg)
            gates={name:all(v[field]>=cfg['minimum_positive_points_per_direction'] for v in support.values()) for name,field in [('raw','raw_positive'),('stable','stable_positive')]}
            scores={name:entry['scores'][old] for name,old in [('nn','old_nn'),('visibility','ordinary_overlap_rgb')]}
            rows.append(dict(support=support,gates=gates,scores=scores))
        choices={}
        for method in ['nn','visibility']:
            choices[method]=min(range(len(rows)),key=lambda i:rows[i]['scores'][method])
            for gate in ['raw','stable']:
                valid=[i for i,row in enumerate(rows) if row['gates'][gate]]
                choices[method+'_'+gate+'_gate']=min(valid,key=lambda i:rows[i]['scores'][method]) if valid else None
        dest=out/path.parent.name;dest.mkdir();save(dest/'predictions.json',dict(pair=pair,choices=choices,rows=rows,pose_pool_sha256=sha(path),GT_read=False))
        runs.append(dict(pair=path.parent.name,poses=len(rows),points_per_pose=len(h.all_xyz)+len(c.all_xyz),elapsed_s=time.perf_counter()-start))
        print('DONE',runs[-1],flush=True)
    save(out/'runs.json',runs)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))


if __name__=='__main__':main()
