"""Meaningful geometry, derivative, role switch, censoring and joint tests."""
import numpy as np
from inspection_decisions.sparse_ray_material_incidence_v1 import region,profiled_factor,hypotheses,polygon_distances,nearest_support_cost,InvalidIncidence,joint_residual,fit_joint_track
from scripts.run_rigid_identity_bounds_v1 import read


def observation(origin,point,outside,depth=3.,epoch=0,ray_id=0,translation=None):
    origin=np.asarray(origin,float);point=np.asarray(point,float);outside=np.asarray(outside,float)
    if translation is not None:point=point+translation;outside=outside+translation
    return {'origin':origin,'ray_inside':(point-origin)/depth,'ray_outside':(outside-origin)/depth,'depth':depth,'outside_depth':30.,'epoch':epoch,'view':0,'ray_id':ray_id}


def checks():
    cfg=read('configs/sparse_ray_material_incidence_v1.json');rng=np.random.default_rng(25210);count=0;maximum=0.;vertex_error=0.;truncations=0;profiled_interior=0
    for truncate in [False,True]:
        for j in range(16):
            x=rng.normal(0,.015,6);obs=observation([-2.,.3,-3.],[-.2,0.,0.],[-.16,.045,0.]);a,k=(0,2) if j%2 else (2,0)
            if truncate:
                from inspection_decisions.sparse_ray_material_incidence_v1 import incidence_data
                data=incidence_data(x,obs,a,k,cfg);obs['outside_depth']=data['ratio']*obs['depth']+.0025
            obj=region(x,obs,a,k,cfg,True);support=obj['vertices'].mean(0)+rng.normal(0,.015,2);r,J,log=profiled_factor(x,obs,a,k,support,cfg);truncations+=log['first_return_truncates_source_interval'];profiled_interior+=log['source_depth_interval'][0]+1e-7<log['source_depth_profiled']<log['source_depth_interval'][1]-1e-7
            for dim in range(8):
                step=np.zeros(8);step[dim]=1e-7;xx=np.r_[x,support];rp=profiled_factor((xx+step)[:6],obs,a,k,(xx+step)[6:],cfg)[0];rm=profiled_factor((xx-step)[:6],obs,a,k,(xx-step)[6:],cfg)[0];fd=(rp-rm)/(2e-7);err=float(np.max(np.abs(fd-J[:,dim])));maximum=max(maximum,err);assert err<3e-4,(truncate,dim,err);count+=1
            for dim in range(6):
                step=np.zeros(6);step[dim]=1e-7;fd=(region(x+step,obs,a,k,cfg)['vertices']-region(x-step,obs,a,k,cfg)['vertices'])/(2e-7);err=float(np.max(np.abs(fd-obj['jacobian'][:,:,dim])));vertex_error=max(vertex_error,err);assert err<3e-5;count+=1
            # A shared support generated inside its correlated polygon gives zero.
            support=obj['vertices'].mean(0);rr,_,_=profiled_factor(x,obs,a,k,support,cfg);assert rr@rr<1e-14;count+=1
    assert truncations>0;count+=1
    # A rectangle bound would incorrectly admit this off-diagonal point.
    P=np.array([[0.,0.],[0.,.1],[1.,1.1],[1.,1.]]);q=np.array([.05,1.05]);assert np.all(q>=P.min(0)) and np.all(q<=P.max(0));assert polygon_distances(P[None],np.tile(q,(4,1))[None])[0]>.5;count+=1
    # Degenerate parallel disjoint lines and isolated points cannot overlap.
    P=np.array([[0.,0.],[1.,0.],[1.,0.],[0.,0.]]);Q=P+[2.,0.];assert abs(polygon_distances(P[None],Q[None])[0]-1)<1e-12;count+=1
    assert abs(polygon_distances(np.zeros((1,4,2)),np.ones((1,4,2)))[0]-np.sqrt(2))<1e-12;count+=1
    # Same actual edge observed from front versus side: no ordered-key identity.
    H=observation([0,0,-3],[-.22,.05,-.24],[-.26,.05,-.24],2.76);C=observation([-3,0,2],[-.24,.05,-.22],[-.24,.05,-.26],2.76,2)
    h=region(np.zeros(6),H,2,0,cfg);c=region(np.zeros(6),C,0,2,cfg);assert h['key']==c['key']==(0,2);assert polygon_distances(h['vertices'][None],c['vertices'][None])[0]==0;count+=1
    invariant=nearest_support_cost([c],[h],1,cfg,False);ordered=nearest_support_cost([c],[h],1,cfg,True);assert invariant['mean_capped_squared_cost']==0 and ordered['mean_capped_squared_cost']==1;count+=1
    # Correctly censored alternative never becomes a free-space factor.
    near=dict(C);near['outside_depth']=1.
    try:region(np.zeros(6),near,0,2,cfg);raise AssertionError('occlusion not censored')
    except InvalidIncidence:count+=1
    # Separate physical supports with identical axes remain distinct regions.
    shifted=dict(H);shifted['origin']=H['origin']+np.array([.5,0,0]);other=region(np.zeros(6),shifted,2,0,cfg);assert polygon_distances(h['vertices'][None],other['vertices'][None])[0]>.4;count+=1
    # Joint pose/edge factor changes along old plane-only depth gauge.
    edge=np.array([-.24,-.24]);true_state=np.r_[np.zeros(12),edge];records=[{'epoch':0,'observation':H,'a':2,'k':0},{'epoch':2,'observation':C,'a':0,'k':2}]
    rr,JJ=joint_residual(true_state,records,cfg);wrong=true_state.copy();wrong[11]=.1;rw,Jw=joint_residual(wrong,records,cfg);assert rr@rr<1e-12 and rw@rw>1.;count+=1
    # Shared support resists a false displacement; independent new support escapes.
    cwrong=region(wrong[6:12],C,0,2,cfg);free=cwrong['vertices'].mean(0);rf,_,_=profiled_factor(wrong[6:12],C,0,2,free,cfg);assert rf@rf<1e-12;count+=1
    for dim in range(14):
        step=np.zeros(14);step[dim]=1e-7;fd=(joint_residual(wrong+step,records,cfg)[0]-joint_residual(wrong-step,records,cfg)[0])/(2e-7);assert np.max(np.abs(fd-Jw[:,dim]))<3e-4;count+=1
    fitted,log=fit_joint_track(wrong,records,[.24,.16],cfg);assert log['final_cost']<float(rw@rw) and log['accepted_steps'];count+=1
    try:joint_residual(wrong,records+[{'epoch':0,'observation':H,'a':0,'k':1}],cfg);raise AssertionError('different coordinate pairs silently merged')
    except ValueError:count+=1
    for state in log['accepted_steps']:
        assert state['after']<state['before']
        for start in [0,6]:assert np.linalg.norm(state['parameters'][start:start+3])<=.24+1e-12 and np.linalg.norm(state['parameters'][start+3:start+6])<=.16+1e-12
        count+=1
    return {'checks':count,'max_profile_jacobian_error':maximum,'max_region_vertex_jacobian_error':vertex_error,'first_return_truncated_cases':truncations,'profiled_depth_interior_cases':profiled_interior,'face_role_switch_cost_invariant':invariant['mean_capped_squared_cost'],'face_role_switch_cost_ordered':ordered['mean_capped_squared_cost'],'conditional_joint_initial_wrong_cost':float(rw@rw),'conditional_joint_final_cost':log['final_cost'],'conditional_joint_accepted_steps':len(log['accepted_steps']),'free_birth_escape_retained_as_control':True}


if __name__=='__main__':print(checks())
