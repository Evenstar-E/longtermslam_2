"""Public-input stationarity probe; separate from frozen predictions."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,setup,dense_records,DensePair,ContinuousTriple,PAIRS,load,fit_pose
from inspection_decisions.material_correspondence_mode_graph_v6 import retract
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);assert json.loads((root/'rich_diagnostics.json').read_text(encoding='utf-8'))['complete'];_,_,_,_,_,_,views,banks,_=setup(c,'rich');dense=dense_records(views);rows=[];payload={}
    @lru_cache(maxsize=128)
    def pair(pe,i,j):
        e,f=PAIRS[pe];return DensePair(banks[e][i],banks[f][j],views[e][i],views[f][j],i,j,dense[e][i],dense[f][j],c)
    for m,col in [('continuous_pair',3),('material_track_shared',6)]:
        p=load(root/f'rich_{m}.npz')
        for i,(j,k) in enumerate(p['target_slots']):
            j,k=int(j),int(k);sc=ContinuousTriple([pair(0,i,j),pair(1,i,k),pair(2,j,k)],c);T=p['transforms'][i];start=float(sc.evaluate(T)[col]);U,path=fit_pose(sc,T,c,col);base=float(sc.evaluate(U)[col]);gradient=[];epsilon=1e-5
            for axis in np.eye(12):gradient.append((sc.evaluate(retract(U,epsilon*axis))[col]-sc.evaluate(retract(U,-epsilon*axis))[col])/(2*epsilon))
            gradient=np.array(gradient);d=-gradient*.005/max(np.linalg.norm(gradient),1e-12);best=base;end=U
            for alpha in [1,.5,.25,.125]:
                Q=retract(U,alpha*d);v=float(sc.evaluate(Q)[col])
                if v<best-1e-10:best=v;end=Q;break
            rows.append(dict(method=m,asset=i,initial_objective=start,additional_geometric_fit_objective=base,full_gradient_probe_objective=best,full_gradient_improves=int(best<base-1e-10),additional_geometry_improves=int(base<start-1e-10),gradient_norm=float(np.linalg.norm(gradient))));payload[f'{m}_{i}_initial']=T;payload[f'{m}_{i}_geometry']=U;payload[f'{m}_{i}_probe']=end
    np.savez_compressed(root/'public_solver_probe.npz',**payload)
    save(root/'public_solver_probe.json',dict(private_labels_read=False,rows=rows,inputs={str(root/f'rich_{m}.npz'):sha(root/f'rich_{m}.npz') for m in ['continuous_pair','material_track_shared']},protocol_sha=sha(root/'frozen_protocol.json'),summary={m:dict(tasks=24,additional_geometry_improves=sum(r['additional_geometry_improves'] for r in rows if r['method']==m),gradient_after_geometry_improves=sum(r['full_gradient_improves'] for r in rows if r['method']==m)) for m in ['continuous_pair','material_track_shared']},scope='All24 rich tasks selected without GT filtering. 3 additional same approximate geometry steps, then one full-objective finite-difference descent/backtracking probe. Not a new primary algorithm or success score; endpoints never inserted into frozen predictions. Energy descent alone cannot establish physical benefit.'))
    print('PUBLIC SOLVER PROBE',sum(r['full_gradient_improves'] for r in rows),'/48 still descend after approximate fit')


if __name__=='__main__':main()
