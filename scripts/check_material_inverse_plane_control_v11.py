import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_inverse_plane_control_v11 import project,SOURCE_DESIGN,SOURCE_PINV
from inspection_decisions.material_oriented_surface_control_v10 import project as legacy
from scripts.run_textured_material_comparison_v1 import save

def main():
    Y,X=np.indices((40,40));K=np.array([[25.,0,20],[0,25.,20],[0,0,1.]]);q=.5+.005*(X-20)+.003*(Y-20);D=1/q;f=dict(C=np.eye(4),K=K,depth=D,depth_valid=np.ones((40,40),bool),anonymous_slot=np.zeros((40,40),int),rgb=np.stack([X/40,Y/40,(X+Y)/80],2)*255);uv=np.array([[20.2,20.3],[18.3,22.1],[21.6,19.7]]);z=1/(.5+.005*(uv[:,0]-20)+.003*(uv[:,1]-20));world=np.c_[(uv[:,0]-20)*z/25,(uv[:,1]-20)*z/25,z];r=project(world,f,0);old=legacy(world,f,0);assert r['surface'].all() and not old['surface'].any();assert np.max(abs(r['gap']))<1e-12;checks=3
    n=-np.array([.005*25,.003*25,.5]);n/=np.linalg.norm(n);assert np.max(abs(r['normal']-n))<1e-12;checks+=1
    z=1/(SOURCE_DESIGN@np.array([.005,.003,.5]));fitted=SOURCE_DESIGN@(SOURCE_PINV@(1/z));assert np.ptp(z)>.025 and np.max(abs(1/fitted-z))<1e-12;checks+=2
    f['depth']=np.ones((40,40))*2;f['depth'][:,21:]+=1;r=project(world,f,0);assert not r['surface'][0] and r['surface'][1:].all();checks+=1
    save(Path('results/material_inverse_plane_check_v11.json'),dict(passed=True,checks=checks,slanted_plane_depth_error_m=float(np.max(abs(1/fitted-z))),meaning='Slant-invariant plane correctness and step rejection; not task efficacy or originality.'));print('INVERSE PLANE CHECK',checks)

if __name__=='__main__':main()
