"""Finite-difference, profile equivalence, censoring and nonfilled-hole checks."""
import numpy as np
from inspection_decisions.edge_conditioned_material_support_v1 import endpoints,profile_intervals,edge_residual,edge_pair_costs,extract_edges
from inspection_decisions.material_supported_continuous_alignment_v1 import world_pose
from scripts.run_rigid_identity_bounds_v1 import read


def feature(a,k,points,step,view=0):
    return {'key':(a,k),'inside':np.asarray(points,float),'outside':np.asarray(points,float)+np.asarray(step),'view':view}


def checks():
    cfg=read('configs/edge_conditioned_material_support_v1.json');cfg.update(focal_px=80.);rng=np.random.default_rng(2732);count=0;maximum=0.
    # A correct, broad common interval carries no information.
    r,J,_=profile_intervals(np.array([0.,.2]),np.array([1.,.8]),np.ones(2),np.zeros((2,12)),np.zeros((2,12)));assert np.all(r==0) and np.all(J==0);count+=1
    # Contradictory interval is retained as two one-sided factors, never dropped.
    cost,b=profile_intervals(np.array([2.]),np.array([0.]),np.ones(1));assert abs(cost-2)<1e-9 and abs(b-1)<1e-9;count+=1
    history=[feature(0,5,[[.24,.01,-.23],[.24,.1,-.23]],[0,0,-.04])]
    ff=[[feature(0,5,[[.24,.01,-.11],[.24,.1,-.11]],[0,0,-.03])],[feature(0,5,[[.24,.01,-.05],[.24,.1,-.05]],[0,0,-.04])]]
    for shared in [False,True]:
        for _ in range(12):
            x=rng.normal(0,.02,12);r,J=edge_residual(x,ff,history,cfg,shared)
            for k in range(12):
                delta=np.zeros(12);delta[k]=1e-6;fd=(edge_residual(x+delta,ff,history,cfg,shared)[0]-edge_residual(x-delta,ff,history,cfg,shared)[0])/(2e-6)
                maximum=max(maximum,float(np.max(np.abs(fd-J[:,k]))));assert np.allclose(fd,J[:,k],atol=2e-5);count+=1
    # Exact matrix profile equals residual profile, with varying per-slot support.
    slots=np.zeros((2,3));features=[[ff[0],[]],[ff[1],ff[0]]];candidates=[[{'slot':j%2,'T':world_pose(rng.normal(0,.02,6),slots[j%2])} for j in range(3)] for _ in range(2)]
    ei,es=edge_pair_costs(candidates,features,slots,history,cfg);assert np.min(es-ei)>-1e-6;count+=1
    for k,l in np.ndindex(ei.shape):
        chosen=[candidates[0][k],candidates[1][l]];from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
        x=np.r_[pose_vector(chosen[0]['T'],slots[chosen[0]['slot']]),pose_vector(chosen[1]['T'],slots[chosen[1]['slot']])]
        for shared,cost in [(False,ei[k,l]),(True,es[k,l])]:
            r,_=edge_residual(x,[features[t][b['slot']] for t,b in enumerate(chosen)],history,cfg,shared);assert abs(r@r-cost)<1e-5;count+=1
    # Measured rectangular plane with a hole and a foreground occluder.
    h,w=64,80;yy,xx=np.mgrid[:h,:w];lab=np.full((h,w),-1);lab[12:52,15:65]=0;lab[27:38,34:46]=-1
    depth=np.where(lab==0,4.,30.);depth[12:52,65:68]=2.;frame={'depth':depth,'C':np.eye(4),'edge_view':0}
    world=np.stack([(xx-(w-1)/2)*depth/80,(yy-(h-1)/2)*depth/80,depth],-1)
    planes=[{'axis':5,'view':0,'normal':np.array([0.,0.,-1]),'centroid':world[lab==0].mean(0),'measured_indices':np.arange((lab==0).sum())}]
    records,log=extract_edges(frame,lab,world,0,np.zeros(3),planes,cfg,21);assert records and log['occlusion_or_unresolved_censored']>0;count+=1
    inside=np.concatenate([f['inside_pixels'] for f in records]);outside=np.concatenate([f['outside_pixels'] for f in records]);assert np.all(lab[inside[:,1],inside[:,0]]==0);assert np.all(depth[outside[:,1],outside[:,0]]>4.03);count+=2
    assert any(np.any((f['outside_pixels'][:,0]>=34)&(f['outside_pixels'][:,0]<46)&(f['outside_pixels'][:,1]>=27)&(f['outside_pixels'][:,1]<38)) for f in records);count+=1
    return {'checks':count,'max_analytic_jacobian_error':maximum,'synthetic_hole_local_segments':len(records),'censored_occluding_rays':log['occlusion_or_unresolved_censored'],'no_surface_filling':True}


if __name__=='__main__':print(checks())
