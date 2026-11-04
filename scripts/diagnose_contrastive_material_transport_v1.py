"""Post-seal GT-reference resolution diagnosis; never creates inference poses."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.contrastive_material_transport_v1 import Model

def main():
    cfg=read('configs/contrastive_material_transport_v1.json');root=Path(cfg['public_root']);fc=read('results/ray_material_full_resolution_v1/effective_configuration.json');meta=read(root/'public.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in meta['clouds']};rows=[]
    for k,pair in enumerate(meta['pairs']):
        a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);GT=b@np.linalg.inv(a)
        for n in [64,256]:
            model=Model(frames[pair['history_stem']],frames[pair['current_stem']],dict(cfg,points=n));scores,diagnosis=model.score(GT);rows.append(dict(pair=f'pair_{k:02d}',points=n,reference_scores=scores,diagnosis=diagnosis))
    pred=read('results/contrastive_material_transport_v1/scores.json');selective=[]
    for method in sorted({r['method'] for r in pred['rows']}):
        rr=[r for r in pred['rows'] if r['method']==method];accepted=[r for r in rr if r['score'] < -1e-12];selective.append(dict(method=method,accepted=len(accepted),abstained=len(rr)-len(accepted),accepted_within_1cm=sum(r['anchor_error_m']<=.01 for r in accepted),scope='POSTHOC zero-evidence abstention interpretation, not a calibrated confidence threshold or a preregistered benefit.'))
    save('results/contrastive_material_transport_v1/diagnosis.json',dict(rows=rows,selective=selective,scope='Reference-only discretization check; no tuned configuration or new test data.'))
    print('HARD_PAIRS',[r for r in rows if r['pair'] in ['pair_07','pair_08']]);print(selective)

if __name__=='__main__':main()
