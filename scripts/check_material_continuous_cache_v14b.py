import numpy as np
from pathlib import Path
from scripts.run_material_continuous_surface_correspondence_v14 import configuration,setup,dense_records,load
from inspection_decisions import material_continuous_surface_correspondence_v14 as original
from inspection_decisions import material_continuous_surface_cache_v14b as fast
from inspection_decisions.material_surface_patch_registration_v9 import retract6
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();d=setup(c,'rich');views,banks=d[6],d[7];dense=dense_records(views);p=load(Path(c['retained_root'])/'rich_cache.npz');i,j,k=map(int,p['triples'][0]);old=[];new=[]
    for (e,f),a,b in zip(original.PAIRS,[i,i,j],[j,k,k]):
        args=(banks[e][a],banks[f][b],views[e][a],views[f][b],a,b,dense[e][a],dense[f][b],c);old.append(original.DensePair(*args));new.append(fast.DensePair(*args))
    A=original.ContinuousTriple(old,c);B=fast.ContinuousTriple(new,c);T=p['transforms'][0,6];U=T.copy();U[1]=retract6(U[1],np.array([.001,0,0,.001,0,0]));batch=np.array([T,U,T]);error=float(np.max(abs(A.score(batch)-B.score(batch))));assert error==0
    ta,pa=original.fit_pose(A,T,c,6);tb,pb=fast.fit_pose(B,T,c,6);assert np.array_equal(ta,tb) and np.array_equal(pa,pb)
    save(Path('results/material_continuous_cache_equivalence_v14b.json'),dict(passed=True,public_only=True,checks=3,max_score_difference=error,fit_transform_equal=True,fit_path_equal=True,scope='Caching KD trees and repeated exact-T scores only. Model, observations, candidate graph and all arithmetic unchanged.'));print('V14 CACHE EQUIVALENCE PASS',error)


if __name__=='__main__':main()
