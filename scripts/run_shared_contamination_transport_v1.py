from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.shared_contamination_transport_v1 import Model,shared_bounds

def main():
    cp=Path('configs/shared_contamination_transport_v1.json');cfg=read(cp);root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);out=Path('results/shared_contamination_transport_v1');out.mkdir(exist_ok=False)
    for p,h in read(pool/'prediction_seal.json')['sha256'].items():assert sha(pool/p)==h
    sources=[Path(__file__),cp,Path('inspection_decisions/shared_contamination_transport_v1.py'),Path('inspection_decisions/contrastive_material_transport_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py')];inputs=[root/'public.json']+list(pool.glob('pair_*/predictions.json'))+[p for p in (root/'public').rglob('*') if p.is_file()]
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},cfg=cfg,GT_read=False))
    fc=read(pool/'effective_configuration.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']}
    for pth in sorted(pool.glob('pair_*/predictions.json')):
        p=read(pth);pair=p['pair'];m=Model(frames[pair['history_stem']],frames[pair['current_stem']],cfg);rows=[]
        for entry in p['pool']:
            scores,d=m.score(np.array(entry['T_current_from_history']));rows.append(dict(scores=scores,diagnosis=d))
        choices={key:max(range(len(rows)),key=lambda i:rows[i]['scores'][key]) for key in rows[0]['scores']};accepted={key:rows[i]['scores'][key]>1e-9 for key,i in choices.items()}
        dest=out/pth.parent.name;dest.mkdir();save(dest/'predictions.json',dict(pair=pair,choices=choices,accepted=accepted,rows=rows,GT_read=False));print('DONE',pth.parent.name,flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))

if __name__=='__main__':main()
