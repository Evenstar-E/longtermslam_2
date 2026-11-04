import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_oriented_surface_control_v10 import project,OrientedScore,orient_banks
from inspection_decisions.material_surface_patch_registration_v9 import bilinear_projection,SurfaceScore,world_points
from scripts.run_textured_material_comparison_v1 import save

def main():
    cp=json.loads(Path('configs/material_oriented_surface_control_v10.json').read_text(encoding='utf-8'));cfg=json.loads(Path(cp['surface_config']).read_text(encoding='utf-8'));cfg.update(cp);Y,X=np.indices((40,40));f=dict(C=np.eye(4),K=np.array([[25.,0,20],[0,25.,20],[0,0,1.]]),depth=1+X*.0002+Y*.0003,depth_valid=np.ones((40,40),bool),anonymous_slot=np.zeros((40,40),int),rgb=np.stack([X/40,Y/40,(X+Y)/80],2)*255);rng=np.random.default_rng(261027);x=rng.uniform(-.2,.2,(40,3));x[:,2]+=1;r=project(x,f,0);old=bilinear_projection(x,f,0,True);n=-old['normal'];n/=np.linalg.norm(n,axis=1)[:,None];error=float(np.max(abs(n-r['normal'])));assert error<1e-12;checks=1
    for field in ['surface','valid','gap','rgb']:assert np.array_equal(r[field],old[field]);checks+=1
    f['depth']=np.ones((40,40));p=np.array([[20,20],[18,20],[22,20],[20,18],[20,22]]);a=dict(xyz=world_points(f,p),rgb=f['rgb'][p[:,1],p[:,0]]/255.,patch_xyz=world_points(f,p)[None],patch_rgb=(f['rgb'][p[:,1],p[:,0]]/255.)[None],patch_frame=np.array([0]));orient_banks([[a]],[[f]],cfg);s=OrientedScore(a,a,[f],[f],0,0,cfg);legacy=SurfaceScore(a,a,[f],[f],0,0,cfg)
    T=np.eye(4);z=s.score(T[None])[0];assert np.allclose(z,0);checks+=1
    b={k:v.copy() for k,v in a.items()};b['patch_normal']*=-1;z=OrientedScore(a,b,[f],[f],0,0,cfg).score(T[None])[0];assert np.allclose(z[:2],0) and np.allclose(z[2:],.5);checks+=1
    b['patch_normal_valid'][:]=False;z=OrientedScore(a,b,[f],[f],0,0,cfg).score(T[None])[0];assert np.allclose(z,0);checks+=1
    transforms=np.tile(np.eye(4),(6,1,1));transforms[:,2,3]=[0,.01,.2,-.2,.5,-.5];new=s.score(transforms);old=legacy.score(transforms)[:,2:];assert np.max(abs(new[:,:2]-old))<1e-12 and (new[:,2]>=new[:,0]-1e-12).all();checks+=2
    assert np.max(abs(new-np.array([s.score(T[None])[0] for T in transforms])))<1e-12;checks+=1
    out=dict(passed=True,checks=checks,target_normal_max_error=error,meaning='Standard physical compatibility correctness, not novelty/task efficacy.');save(Path('results/material_oriented_surface_check_v10.json'),out);print(json.dumps(out))

if __name__=='__main__':main()
