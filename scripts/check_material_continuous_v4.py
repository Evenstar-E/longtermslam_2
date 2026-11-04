"""Mechanism tests: measured support, off-grid decode, holes, finite training."""
import numpy as np
import torch
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher,closest_on_triangles,decode_local,surface_grids


def main():
    torch.set_num_threads(1);torch.manual_seed(31)
    tri=np.array([[[0.,0,0],[1.,0,0],[0.,1,0]]])
    assert np.allclose(closest_on_triangles(np.array([.2,.3,1]),tri),[.2,.3,0])
    assert np.allclose(closest_on_triangles(np.array([1.,1,0]),tri),[.5,.5,0])
    assert closest_on_triangles(np.zeros(3),np.zeros((0,3,3))) is None
    g=np.linspace(-.06,.06,12);xx,yy=np.meshgrid(g,g);xyz=np.c_[xx.ravel(),yy.ravel(),np.zeros(144)]
    triangles=[]
    for y in range(11):
        for x in range(11):
            k=y*12+x;triangles.extend([[k,k+1,k+12],[k+1,k+13,k+12]])
    grid=dict(valid=np.ones((1,144),bool),xyz=xyz[None],triangles=np.array(triangles),tri_valid=np.ones((1,242),bool))
    wanted=np.array([.002,.003,0]);score=-((xyz-wanted)**2).sum(1)/.008**2
    point,_=decode_local(score,grid,0,False);continuous,_=decode_local(score,grid,0,True)
    assert np.linalg.norm(continuous-wanted)<np.linalg.norm(point-wanted)
    grid['tri_valid'][:]=False;fallback,_=decode_local(score,grid,0,True);assert np.array_equal(fallback,point)
    grid['valid'][:]=False;assert decode_local(score,grid,0,True)==(None,0.)
    # Actual measured grid cannot bridge a depth jump or another anonymous slot.
    h=w=64;frame=dict(depth=np.ones((h,w))*2,depth_valid=np.ones((h,w),bool),anonymous_slot=np.zeros((h,w),int),C=np.eye(4),K=np.array([[200.,0,32],[0,200.,32],[0,0,1]]))
    groups=[dict(xyz=np.zeros((0,3)),frame=np.zeros(0,int),uv=np.zeros((0,2))) for _ in range(24)]
    groups[0]=dict(xyz=np.array([[0.,0,2]]),frame=np.array([0]),uv=np.array([[32.,32.]]))
    cfg=dict(patch_size=12,normal_window_px=3);v4=dict(surface_radius_m=.06,surface_edge_limit_m=.04)
    normal=surface_grids([frame],groups,cfg,v4)[0];assert normal['tri_valid'].sum()>0
    frame['anonymous_slot'][:,32:]=1
    cut=surface_grids([frame],groups,cfg,v4)[0]
    assert cut['valid'].sum()<normal['valid'].sum() and cut['tri_valid'].sum()<normal['tri_valid'].sum()
    assert (cut['valid'][0,cut['triangles'][cut['tri_valid'][0]]]).all()
    model=SurfaceMatcher();a=torch.randn(4,10,12,12);b=torch.randn_like(a);mask=torch.ones(4,144,dtype=torch.bool);mask[0]=False
    scores=model(a,b,mask);assert torch.isfinite(scores).all()
    target=torch.tensor([144,0,20,100]);loss=torch.nn.functional.cross_entropy(scores,target);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert scores[0].argmax()==144
    print('PASS triangle projection, observed off-grid improvement, hole fallback, slot/depth support mask, null class and finite gradients',flush=True)


if __name__=='__main__':main()
