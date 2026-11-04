"""Independent diagnostic and shared-factor capacity checks."""
import numpy as np
from pathlib import Path
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import persistence_factors
from inspection_decisions.material_surface_patch_registration_v9 import retract6
from scripts.diagnose_material_full_propagation_v12 import batch_error
from scripts.diagnose_material_local_information_v12 import spectra
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();rng=np.random.default_rng(12);x=rng.normal(size=(12,3));y1=x+.02;y2=x-.03
    T=np.array([[retract6(np.eye(4),rng.normal(size=6)*.02) for _ in range(2)] for _ in range(5)])
    error=float(np.max(abs(batch_error(T,x,y1,y2)-[physical_error(*t,x,y1,y2) for t in T])));assert error<1e-12
    _,rank=spectra(np.c_[np.eye(3),np.zeros((3,3))]);assert rank==3
    x,y=np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,101));value=persistence_factors(x.reshape(-1,1),y.reshape(-1,1),cfg);delta=cfg['material_factor_weight']*(value[:,0]-value[:,1]);pi=cfg['shared_prior'];eta=cfg['material_null_likelihood']
    bound=cfg['material_factor_weight']*np.log((pi+(1-pi)*eta**2)/(pi+(1-pi)*eta)**2)/(-2*np.log(eta));assert delta.min()>-1e-14 and abs(delta.max()-bound)<1e-12
    save(Path('results/material_inverse_plane_pose_math_check_v12.json'),dict(passed=True,checks=3,capacity_grid_points=10201,batch_error_max=error,maximum_shared_rebate=float(bound),capacity_grid_max_error=float(abs(delta.max()-bound)),scope='Independent diagnostics and algebra correctness, not task efficacy.'));print('V12 MATH CHECKS 3; CAPACITY',bound)


if __name__=='__main__':main()
