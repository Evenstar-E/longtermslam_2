"""Analytic world-point derivative on actual measured triangle branches."""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from inspection_decisions.material_exposure_derivatives_v15 import exposure_loss_gradient
from scripts.run_material_continuous_surface_correspondence_v14c import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();Y,X=np.indices((32,32));K=np.array([[100.,0,16],[0,100.,16],[0,0,1.]]);C=np.eye(4);C[:3,:3]=Rotation.from_rotvec([.2,-.15,.1]).as_matrix();C[:3,3]=[.3,-.4,.2]
    d=1/(.5+.002*(X-16)+.001*(Y-16));f=dict(C=C,K=K,depth=d,depth_valid=np.ones((32,32),bool),anonymous_slot=np.zeros((32,32),int),rgb=np.full((32,32,3),127.5));rng=np.random.default_rng(15);uv=rng.uniform(8,22,size=(100,2));z=1/(.5+.002*(uv[:,0]-16)+.001*(uv[:,1]-16))-.05
    pc=np.c_[(uv[:,0]-16)*z/100,(uv[:,1]-16)*z/100,z];p=pc@C[:3,:3].T+C[:3,3];r=exposure_loss_gradient(p,f,0,cfg);assert r['active'].all();maximum=0.
    for k in range(3):
        step=np.eye(3)[k]*1e-6;a=exposure_loss_gradient(p+step,f,0,cfg)['loss'];b=exposure_loss_gradient(p-step,f,0,cfg)['loss'];maximum=max(maximum,float(np.max(abs((a-b)/2e-6-r['gradient'][:,k]))))
    assert maximum<1e-6,maximum
    save(Path('results/material_exposure_derivative_checks_v15.json'),dict(passed=True,points=100,derivative_components=300,max_derivative_error=maximum,scope='Exact derivative only within unchanged measurement/owner/triangle/positive-gap branch. Not integrated into V14c optimizer, not a task gain or an originality claim.'))
    print('EXPOSURE DERIVATIVES',maximum)


if __name__=='__main__':main()
