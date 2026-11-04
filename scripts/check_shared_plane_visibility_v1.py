"""Check sign-safe inequality conversion on measured planar-patch polytopes."""
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.ray_conditioned_plane_polytope_v1 import fit_patch,intersection_bounds
from inspection_decisions.shared_plane_visibility_v1 import visibility_halfspaces,feasible_shared_plane


def main():
    root=Path('results/first_return_material_surface_v1');cfg=read(root/'config.json');rows=[];rng=np.random.default_rng(85012)
    for control in read(root/'plane_polytope_prototype.json')['controls']:
        path=root/control['scene']/control['frame'];frame=np.load(path);x,y=control['pixel']
        patch=fit_patch(frame['depth'][y-1:y+2,x-1:x+2],[x,y],cfg['focal_px'],[(cfg['width']-1)/2,(cfg['height']-1)/2],cfg['depth_noise_bound_m'])
        pr=patch['principal'];r=np.array([[(x-pr[0])/cfg['focal_px'],(y-pr[1])/cfg['focal_px'],1.]])
        for o,direction in [(np.zeros(3),r),(40*r[0],-r)]:
            bounds=intersection_bounds(patch,o,direction);assert bounds['valid'][0]
            # Deliberately sweep synthetic test depths around the exact interval,
            # testing algebra only; not tuning an inference hyperparameter.
            for depth in np.linspace(bounds['lower'][0]-.01,bounds['upper'][0]+.01,11):
                factor=visibility_halfspaces(patch,o,direction,[depth],0.)
                vv=patch['coefficients'];direct=(1-patch['normals']@o)/(patch['normals']@direction[0])>=depth-1e-10
                linear=np.all(vv@factor['A'].T<=factor['b']+1e-12,axis=1)
                assert np.array_equal(direct,linear)
                ans=feasible_shared_plane(patch,[factor])
                expected=bool(bounds['upper'][0]>=depth-1e-9);assert ans['feasible']==expected
                rows.append({'depth':float(depth),'feasible':expected})
    save(root/'shared_plane_factor_checks.json',{'status':'PASS','checks':len(rows)*2,'patches':len(read(root/'plane_polytope_prototype.json')['controls']),
         'full_pipeline_run':False,'uses_evaluation_truth':False,'sources':{p:sha(p) for p in [__file__,'inspection_decisions/shared_plane_visibility_v1.py']}})
    print({'status':'PASS','checks':len(rows)*2})


if __name__=='__main__':main()
