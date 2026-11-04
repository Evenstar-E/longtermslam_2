"""Independent bilinear derivatives, censoring, patch support and fit descent."""
import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_surface_patch_registration_v9 import *
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=json.loads(Path('configs/material_surface_patch_registration_v9.json').read_text(encoding='utf-8'));Y,X=np.indices((40,40));rgb=np.stack([X/40,Y/40,(X+Y)/80],2)*255;f=dict(C=np.eye(4),K=np.array([[25.,0,20],[0,25.,20],[0,0,1.]]),depth=1+X*.0002+Y*.0003,depth_valid=np.ones((40,40),bool),anonymous_slot=np.zeros((40,40),int),rgb=rgb);rng=np.random.default_rng(261129);x=rng.uniform(-.25,.25,(20,3));x[:,2]=rng.uniform(.95,1.05,20);r=bilinear_projection(x,f,0,True);error=0.;checks=0
    for k in range(3):
        d=np.eye(3)[k]*1e-6;a=bilinear_projection(x+d,f,0);b=bilinear_projection(x-d,f,0);color=(a['rgb']-b['rgb'])/2e-6;gap=(a['gap']-b['gap'])/2e-6;error=max(error,float(np.max(abs(color-r['rgbJ'][:,:,k]))),float(np.max(abs(gap+r['normal'][:,k]))));checks+=2
    assert error<1e-7
    f['depth']=np.ones((40,40));pix=np.array([[20,20],[18,20],[22,20],[20,18],[20,22]]);xyz=world_points(f,pix)[None];color=f['rgb'][pix[:,1],pix[:,0]][None]/255.;a,b,free=projection_terms(xyz[None],color[None],[f],0,cfg);assert np.allclose(a,1) and np.allclose(b,1) and np.allclose(free,0);checks+=3
    for dz,positive in [(.2,False),(-.2,True)]:
        y=xyz.copy();y[:,:,2]+=dz;a,b,free=projection_terms(y[None],color[None],[f],0,cfg);assert a[0]==0 and b[0]==0 and bool(free[0]>0)==positive;checks+=3
    # Translation Jacobian in both directions, independent finite differences.
    T=retract6(np.eye(4),rng.normal(size=6)*.1);points=rng.normal(size=(8,3));z=apply(inverse(T),points);J=-np.einsum('ij,njk->nik',T[:3,:3].T,np.concatenate([-skew(points),np.broadcast_to(np.eye(3),(len(points),3,3))],2));jerr=0.
    for k in range(6):
        d=np.eye(6)[k]*1e-6;num=(apply(inverse(retract6(T,d)),points)-apply(inverse(retract6(T,-d)),points))/2e-6;jerr=max(jerr,float(np.max(abs(num-J[:,:,k]))));checks+=1
    assert jerr<1e-7
    record=dict(xyz=xyz.reshape(-1,3),rgb=color.reshape(-1,3),patch_xyz=xyz,patch_rgb=color);s=SurfaceScore(record,record,[f],[f],0,0,cfg);initial=np.eye(4);initial[0,3]=.01;U,path=projective_fit(s,initial,cfg);assert np.max(np.diff(path),initial=0)<=1e-12 and path[-1]<=path[0];checks+=1
    out=dict(passed=True,checks=checks,bilinear_derivative_max_error=error,inverse_pose_derivative_max_error=jerr,fit_path=path.tolist(),meaning='Correctness, not task efficacy or novelty');save(Path('results/material_surface_patch_check_v9.json'),out);print(json.dumps(out))

if __name__=='__main__':main()
