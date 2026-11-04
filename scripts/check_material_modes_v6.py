import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.material_correspondence_mode_graph_v6 import residual_jacobian,retract,joint_fit,pair_modes,material_tracks
from inspection_decisions.textured_material_matching_v1 import apply

def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));rng=np.random.default_rng(261031);x0=rng.normal(size=(20,3))*.15+[2,1,0]
    T=np.tile(np.eye(4),(2,1,1));T[0,:3,:3]=Rotation.from_rotvec([.1,.05,-.03]).as_matrix();T[0,:3,3]=[1,-2,.1];T[1,:3,:3]=Rotation.from_rotvec([-.08,.03,.1]).as_matrix();T[1,:3,3]=[-1,1,.1]
    x=np.stack([x0,apply(T[0],x0)+rng.normal(size=x0.shape)*.001,apply(T[1],x0)+rng.normal(size=x0.shape)*.001],axis=1);r,J=residual_jacobian(T,x);maximum=0.
    for k in range(12):
        d=np.zeros(12);d[k]=1e-6;fd=(residual_jacobian(retract(T,d),x)[0]-residual_jacobian(retract(T,-d),x)[0])/2e-6;maximum=max(maximum,float(np.max(abs(fd-J[:,:,:,k]))))
    assert maximum<1e-7,maximum
    noisy=retract(T,rng.normal(size=12)*.01);fit,path=joint_fit(noisy,x,np.ones(20),cfg);assert np.diff(path).max()<=1e-14 and path[-1]<path[0]
    assert np.linalg.norm(fit-T)<np.linalg.norm(noisy-T)
    edge=dict(x=x0,y=x[:,1],source=np.arange(20),target=np.arange(20)+100,weight=np.ones(20));a=pair_modes(edge,cfg,'ransac',1);b=pair_modes(edge,cfg,'graph',1)
    assert a and b and np.linalg.norm(a[0]-T[0])<.03 and np.linalg.norm(b[0]-T[0])<.03
    # Cycles must be actual observation links, never pose-loop tautologies.
    blank=lambda:dict(x=np.zeros((0,3)),y=np.zeros((0,3)),source=np.zeros(0,int),target=np.zeros(0,int),weight=np.zeros(0))
    edges=[[blank() for _ in range(576)] for _ in range(3)]
    edges[0][0]=edge;edges[1][0]=dict(x=x0,y=x[:,2],source=np.arange(20),target=np.arange(20)+200,weight=np.ones(20));edges[2][0]=dict(x=x[:,1],y=x[:,2],source=np.arange(20)+100,target=np.arange(20)+200,weight=np.ones(20))
    tracks,_=material_tracks(edges,cfg);assert len(tracks[0,0,0]['x'])==20
    edges[2][0]=blank();assert not material_tracks(edges,cfg)[0]
    print('PASS 12 independent joint Jacobian columns, robust descent, native pair rigid modes, actual observation-cycle presence/absence; maxJ',maximum,flush=True)

if __name__=='__main__':main()
