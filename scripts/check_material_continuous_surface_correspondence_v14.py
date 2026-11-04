import numpy as np
from pathlib import Path
from inspection_decisions.material_continuous_surface_correspondence_v14 import measured_field,photo_kernel,joint_photo,solve_offset
from scripts.run_material_continuous_surface_correspondence_v14 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();Y,X=np.indices((32,32));K=np.array([[100.,0,16],[0,100.,16],[0,0,1.]])
    depth=1/(.5+.002*(X-16)+.001*(Y-16));rgb=np.stack([X/32,Y/32,(X+Y)/64],-1)*255;f=dict(C=np.eye(4),K=K,depth=depth,depth_valid=np.ones((32,32),bool),anonymous_slot=np.zeros((32,32),int),rgb=rgb)
    uv=np.array([[14.2,15.3],[17.8,16.6]]);r=measured_field(uv,f,0,cfg);expected=1/(.5+.002*(uv[:,0]-16)+.001*(uv[:,1]-16));assert r['surface'].all() and np.max(abs(r['depth']-expected))<1e-12;checks=1;maximum=0.
    for key,jac in [('xyz','xyzJ'),('rgb','rgbJ')]:
        for k in range(2):
            delta=np.eye(2)[k]*1e-5;a=measured_field(uv+delta,f,0,cfg)[key];b=measured_field(uv-delta,f,0,cfg)[key];error=float(np.max(abs((a-b)/2e-5-r[jac][...,k])));assert error<1e-8,(key,error);maximum=max(maximum,error);checks+=1
    # Mixed-owner cell must retain its actual integer center, not reject it.
    g={k:v.copy() for k,v in f.items()};g['anonymous_slot'][15,15]=1;q=measured_field(np.array([[14.,15.]]),g,0,cfg);assert q['surface'][0] and abs(q['depth'][0]-depth[15,14])<1e-12;checks+=1
    a=np.array([[.1,.2,-.1],[.3,-.1,.2]]);b=np.array([[.2,.1,.1],[-.2,.2,-.1]]);ind=joint_photo(a,b,cfg,False);assert np.max(abs(ind-photo_kernel(a,cfg)*photo_kernel(b,cfg)))<1e-14;checks+=1
    shared=joint_photo(a,b,cfg,True);assert (shared>=0).all() and (shared<=1).all();checks+=1
    zero=np.zeros((1,3));same=joint_photo(zero,zero,cfg,True);assert same[0]>joint_photo(zero,zero,cfg,False)[0];checks+=1
    # The free ray cannot change merely because the target correspondence moves.
    p=r['xyz'].copy();p[:,2]-=.03;color=r['rgb']+.05;fixed=solve_offset(p,color,f,0,cfg,False);optimized=solve_offset(p,color,f,0,cfg,True);assert np.array_equal(fixed['original_gap'],optimized['original_gap']);checks+=1
    assert np.max(np.linalg.norm(optimized['offset'],axis=-1))<=cfg['offset_limit_px']+1e-12;checks+=1
    save(Path('results/material_continuous_surface_checks_v14.json'),dict(passed=True,checks=checks,max_field_derivative_error=maximum,scope='Field, native fallback, Gaussian normalization, original-ray and offset bound correctness. Not task gain.'));print('V14 CHECKS',checks,maximum)


if __name__=='__main__':main()
