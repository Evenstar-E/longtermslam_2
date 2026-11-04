"""Freeze all inference before evaluation; reuse poses, never annotation poses."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.contrastive_material_transport_v1 import Model,evidence

def main():
    cfgpath=Path('configs/contrastive_material_transport_v1.json');cfg=read(cfgpath);root=Path(cfg['public_root']);poolroot=Path(cfg['pool_root']);out=Path('results/contrastive_material_transport_v1');out.mkdir(exist_ok=False)
    for name,h in read(poolroot/'prediction_seal.json')['sha256'].items():assert sha(poolroot/name)==h
    files=[Path(__file__),cfgpath,Path('inspection_decisions/contrastive_material_transport_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py')]
    inputs=[root/'public.json']+[p for p in (root/'public').rglob('*') if p.is_file()]+list(poolroot.glob('pair_*/predictions.json'))
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in files},inputs={str(p):sha(p) for p in inputs},cfg=cfg,GT_read=False))
    # True coupled effect: two witnesses compete for one material witness.
    assert evidence(np.array([[1.,0.],[1.,0.]]),True)<evidence(np.array([[1.,0.],[1.,0.]]),False)
    assert evidence(np.eye(2),True)==evidence(np.eye(2),False)
    assert evidence(np.zeros((2,2)),True)==0
    fc=read('results/ray_material_full_resolution_v1/effective_configuration.json');meta=read(root/'public.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in meta['clouds']}
    for pth in sorted(poolroot.glob('pair_*/predictions.json')):
        p=read(pth);pair=p['pair'];model=Model(frames[pair['history_stem']],frames[pair['current_stem']],cfg);rows=[]
        for e in p['pool']:
            s,d=model.score(np.array(e['T_current_from_history']));rows.append(dict(scores=s,diagnosis=d))
        methods=list(rows[0]['scores']);choices={m:min(range(len(rows)),key=lambda i:rows[i]['scores'][m]) for m in methods}
        dest=out/pth.parent.name;dest.mkdir();save(dest/'predictions.json',dict(pair=pair,choices=choices,rows=rows,pose_pool_sha256=sha(pth),GT_read=False));print('DONE',pth.parent.name,len(rows),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))

if __name__=='__main__':main()
