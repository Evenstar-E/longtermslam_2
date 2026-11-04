import numpy as np
from types import SimpleNamespace
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from inspection_decisions.material_surface_distribution_matching_v15 import compatibility,LearnedPair
from scripts.prepare_material_surface_fields_v15 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();rng=np.random.default_rng(15);x=rng.normal(size=(12,3))*.05;src=np.arange(12);m=dict(source=src,point=x.copy(),mean=x.copy(),cov=np.zeros((12,3,3)),mass=np.ones(12),cosine=np.ones(12));a=dict(xyz=x)
    A,_=compatibility(a,m,c,False);B,_=compatibility(a,m,c,True);assert np.array_equal(A,B);checks=1
    R=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();m['cov']=np.tile(np.diag([.001,.002,.003])**2,(12,1,1));B,_=compatibility(a,m,c,True);rot={**m,'point':x@R.T,'mean':x@R.T,'cov':R@m['cov']@R.T};C,_=compatibility(dict(xyz=x@R.T),rot,c,True);err=float(np.max(abs(B-C)));assert err<1e-6;checks+=1
    desc=np.ones((12,64))/8;a.update(desc=desc,normalizer=np.ones(12));b=dict(xyz=x.copy(),desc=desc.copy(),area=np.ones(12)/12);p=SimpleNamespace(a=a,b=a,da=b,db=b,ta=cKDTree(x),tb=cKDTree(x));pair=LearnedPair(p,c)
    for shift in [0.,.01,.1]:
        T=np.eye(4);T[0,3]=shift;v=pair.feature_values(T);assert abs(v[1])<1e-12,v;checks+=1
    # Splitting identical measurements while preserving total area mass is neutral.
    T=np.eye(4);T[0,3]=.005;before=pair.feature_values(T);dd={k:np.repeat(v,2,axis=0) for k,v in b.items()};dd['area']/=2;p.db=dd;p.tb=cKDTree(dd['xyz']);q=LearnedPair(p,c);assert np.max(abs(q.feature_values(T)-before))<1e-12;checks+=1
    save(Path('results/material_surface_distribution_checks_v15.json'),dict(passed=True,checks=checks,rotation_compatibility_max_error=err,scope='Zero covariance limit, distance covariance rotation, uniform-feature neutral marginal, fixed-area duplicate split. Not task gain or originality.'));print('V15 CHECKS',checks,err)


if __name__=='__main__':main()
