import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from inspection_decisions.material_surface_distribution_v15b import dense_modes,RegionPair
from scripts.prepare_material_surface_fields_v15 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();c.update(json.loads(Path('configs/material_surface_distribution_v15b.json').read_text(encoding='utf-8')));rng=np.random.default_rng(1515);xyz=rng.uniform(-.08,.08,(160,3));desc=np.ones((160,64),np.float32)/8;b=dict(xyz=xyz,desc=desc,area=np.ones(160)/160);a=dict(xyz=xyz[:7],desc=desc[:7],normalizer=np.ones(7));m=dense_modes(a,b,c);p=SimpleNamespace(a=a,b=a);rp=RegionPair(p,m,m,c);checks=0
    for kind in range(3):
        assert abs(rp.one_mixture(a['xyz']+.01,a,m,rp.precision[0],rp.prefactor[0],kind))<1e-12;checks+=1
    zero={**m,'cov':np.zeros_like(m['cov']),'prior_cov':np.zeros_like(m['cov'])};z=RegionPair(p,zero,zero,c);x=a['xyz']+.01
    assert abs(z.one_mixture(x,a,zero,z.precision[0],z.prefactor[0],1)-z.one_mixture(x,a,zero,z.precision[0],z.prefactor[0],2))<1e-12;checks+=1
    for key in ['prior','mass']:assert np.max(np.bincount(m['source'],weights=m[key]))<=1+1e-12;checks+=1
    duplicated={k:np.repeat(v,2,axis=0) for k,v in b.items() if k!='_tree'};duplicated['area']/=2;d=dense_modes(a,duplicated,c)
    for key in ['mass','prior','mean','prior_mean','cov','prior_cov']:assert np.max(abs(m[key]-d[key]))<1e-12,key;checks+=1
    sigma=c['marginal_sigma_m'];C=np.diag(np.array([.004,.008,.001])**2);r=np.array([.02,.01,.004]);S=sigma*sigma*np.eye(3)+C;exact=sigma**3/np.sqrt(np.linalg.det(S))*np.exp(-.5*r@np.linalg.inv(S)@r);samples=rng.multivariate_normal(np.zeros(3),C,size=200000);mc=np.exp(-.5*np.sum((samples-r)**2,axis=1)/sigma**2).mean();assert abs(exact-mc)<.002;checks+=1
    save(Path('results/material_surface_region_checks_v15b.json'),dict(passed=True,checks=checks,gaussian_convolution_exact=float(exact),monte_carlo=float(mc),convolution_error=float(abs(exact-mc)),scope='Disjoint posterior/prior masses, broad residual component, uniform-neutral likelihood ratio, zero covariance limit and fixed-area splitting; Gaussian moment exact only for Gaussian region approximation. Not efficacy/novelty.'));print('V15b REGION CHECKS',checks,abs(exact-mc))


if __name__=='__main__':main()
