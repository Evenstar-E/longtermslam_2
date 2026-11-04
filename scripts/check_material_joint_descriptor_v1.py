"""Independent geometry and learning-boundary checks, not performance claims."""
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_joint_descriptor_v1 import metric_patches,MaterialDescriptor,training_loss
from scripts.build_material_descriptor_train_cache_v1 import training_scene


def plane(distance):
    n=160;f=180.;y,x=np.mgrid[:n,:n];xx=(x-(n-1)/2)*distance/f;yy=(y-(n-1)/2)*distance/f
    value=120+45*np.sin(xx*20)+25*np.cos(yy*32)
    rgb=np.stack([value,value*.8+15,value*.6+35],axis=-1).clip(0,255).astype(np.uint8)
    C=np.eye(4);C[:3,3]=[0,0,-distance]
    return dict(rgb=rgb,depth=np.full((n,n),distance),depth_valid=np.ones((n,n),bool),
        K=np.array([[f,0,(n-1)/2],[0,f,(n-1)/2],[0,0,1.]]),C=C,anonymous_slot=np.zeros((n,n),int))


def main():
    cfg=json.loads(Path('configs/material_joint_descriptor_v1.json').read_text());torch.set_num_threads(2);checks=0
    a=plane(2.);b=plane(4.);uv=np.array([[79.5,79.5]])
    p,x=metric_patches(a,uv,cfg);q,y=metric_patches(b,uv,cfg)
    assert np.max(abs(p-q))<.08;checks+=1
    a['C'][:3,3]+=[12,-9,5];p2,x2=metric_patches(a,uv,cfg)
    assert np.max(abs(p-p2))<1e-5;checks+=1
    assert np.max(abs(x2-x-[12,-9,5]))<1e-10;checks+=1
    try:training_scene(Path(cfg['dataset'])/'test_000')
    except ValueError:checks+=1
    else:raise AssertionError('Test supervision accepted by training boundary')
    torch.manual_seed(3);model=MaterialDescriptor();source=torch.randn(8,10,12,12);target=torch.randn(12,10,12,12)
    visible=torch.tensor([True]*6+[False]*2);pos=torch.arange(8);changed=torch.tensor([False,True]*4)
    near=torch.zeros(8,12,dtype=torch.bool)
    loss,stats=training_loss(model,source,target,visible,pos,changed,near)
    loss.backward();assert np.isfinite(list(stats.values())).all();checks+=1
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);checks+=1
    z=model(source);assert torch.max(abs(torch.linalg.vector_norm(z,dim=1)-1))<1e-6;checks+=8
    # Appearance-change supervision must not alter positive physical labels.
    _,s1=training_loss(model,source,target,visible,pos,torch.zeros_like(changed),near)
    assert abs(s1['correspondence']-stats['correspondence'])<1e-7;checks+=1
    print('PASS',checks,'checks; cross-scale metric patch maxdiff',float(abs(p-q).max()),flush=True)


if __name__=='__main__':main()
