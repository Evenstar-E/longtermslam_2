"""Source uncertainty and shared-marginal mathematical control checks."""
import numpy as np
from pathlib import Path
from inspection_decisions.material_continuous_surface_correspondence_v14 import joint_photo,photo_kernel
from scripts.run_material_continuous_surface_correspondence_v14 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();rng=np.random.default_rng(14);a=rng.normal(0,.3,(100,3));b=rng.normal(0,.3,(100,3));zero=dict(c,source_color_sigma=0.)
    error=float(np.max(abs(joint_photo(a,b,zero,False)-joint_photo(a,b,zero,True))));assert error<1e-14
    r=np.ones((1,3))*.4;same=joint_photo(r,r,c,True)[0];ind=joint_photo(r,r,c,False)[0];opposite=joint_photo(r,-r,c,True)[0];assert same>ind>opposite
    prod=photo_kernel(a,c)*photo_kernel(b,c);assert np.max(abs(prod-joint_photo(a,b,c,False)))<1e-14
    save(Path('results/material_continuous_covariance_checks_v14.json'),dict(passed=True,checks=3,zero_source_sigma_max_model_difference=error,same_sign_shared=float(same),independent=float(ind),opposite_sign_shared=float(opposite),scope='Correct Gaussian covariance/marginalization and limiting case only; not task efficacy or originality.'));print('V14 COVARIANCE CHECKS 3')


if __name__=='__main__':main()
