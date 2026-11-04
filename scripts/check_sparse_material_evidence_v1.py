"""No-GT equation, assignment and actual-observation implementation checks."""
import itertools,pickle,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.run_rigid_identity_bounds_v1 import read
from scripts.run_sparse_ray_material_incidence_v3 import save
from inspection_decisions import sparse_material_evidence_v1 as m
from inspection_decisions.typed_material_incidence_graph_v1 import JointProblem


def checks():
    torch.set_num_threads(1);cfg=read('results/typed_material_incidence_graph_v2/config.json');cfg.update(read('configs/sparse_material_evidence_v1.json'));cfg['evidence_temperature']=cfg['normal_residual_scale_m']**2+cfg['depth_noise_bound_m']**2;tau=cfg['evidence_temperature'];rng=np.random.default_rng(20261025);E1=rng.uniform(0,.001,(5,16));E2=rng.uniform(0,.001,(7,16));W=m.log_weights(rng.uniform(0,.0001,16),tau);U1=rng.random((5,16,9))>.5;U2=rng.random((7,16,9))>.5;maximum=0.;n=0
    for method in cfg['methods']:
        C=m.pair_costs(E1,E2,W,tau,method,U1,U2)
        for a,b in itertools.product(range(5),range(7)):
            e=E1[a]+E2[b]-tau*W
            if method=='profile_mdl_ray':e+=tau*np.log(2)*np.logical_or(U1[a],U2[b]).sum(1)
            expect=E1[a].min() # overwritten in each branch
            if method in ['joint_ray','joint_line']:expect=-tau*logsumexp(-e/tau)
            elif method=='independent_ray':expect=-tau*(logsumexp(W-E1[a]/tau)+logsumexp(W-E2[b]/tau))
            else:expect=e.min()
            maximum=max(maximum,abs(expect-C[a,b]));assert abs(expect-C[a,b])<1e-12;n+=1
    C=m.pair_costs(E1,E2,W,tau,'joint_ray');repeat=m.pair_costs(np.repeat(E1,2,1),np.repeat(E2,2,1),np.repeat(W-np.log(2),2),tau,'joint_ray');assert np.max(abs(C-repeat))<1e-12;n+=1
    # Shared state makes the two-epoch identity cost nonseparable.
    simple=np.array([[0.,1.],[1.,0.]]);J=m.pair_costs(simple,simple,np.log([.5,.5]),.1,'joint_ray');assert abs(J[0,0]+J[1,1]-J[0,1]-J[1,0])>1.;n+=1
    for _ in range(12):
        C=rng.random((3,3,3));starts=[(a,b) for a in itertools.permutations(range(3)) for b in itertools.permutations(range(3))];a,b,log=m.assign(C,starts);exact=min(sum(C[i,aa[i],bb[i]] for i in range(3)) for aa,bb in starts);assert abs(log['energy']-exact)<1e-10;assert log['LP_lower_bound']<=exact+1e-10;n+=2
    parent=Path(cfg['parent_root']);tag='aligned_s270100_h1'
    with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
    graph,states=m.material_states(data[0][0],cfg,27010000);model=m.PeriodModel(data[1][0],graph,cfg);D=torch.tensor(states,dtype=m.DT);gradmax=0.;costmax=0.
    for _ in range(3):
        x=rng.normal(0,.01,6);out=model.evaluate(torch.tensor(x,dtype=m.DT),D,('ray','line'),True)
        for mode in ['ray','line']:
            for l in range(16):
                p=JointProblem([data[1][0]],graph,np.r_[x,np.zeros(6)],[.24,.16],cfg,'line_joint' if mode=='line' else 'typed_ray');p.free_faces=False;p.face0=states[l];p.size=p.depth_start=12;r,J,_=p.evaluate(np.r_[x,np.zeros(6)],True);err=abs(float(r@r)-float(out[mode][0][l]));costmax=max(costmax,err);assert err<1e-12;n+=1
            xx=torch.tensor(x,dtype=m.DT,requires_grad=True);value=model.evaluate(xx,D,(mode,))[mode][0][0];value.backward();r,J,_=p.evaluate(np.r_[x,np.zeros(6)],True) # p currently state15; rebuild state0 below
            p.face0=states[0];r,J,_=p.evaluate(np.r_[x,np.zeros(6)],True);g=2*np.asarray(J.T@r).ravel()[:6];err=np.max(abs(g-xx.grad.numpy()));gradmax=max(gradmax,float(err));assert err<1e-9;n+=1
    xx=torch.zeros(6,dtype=m.DT,requires_grad=True);model.evaluate(xx,D)["ray"][0].sum().backward();assert torch.isfinite(xx.grad).all();n+=1
    start=time.perf_counter()
    with torch.inference_mode():
        for _ in range(100):model.evaluate(torch.zeros(6,dtype=m.DT),D,('ray','line'),True)
    return {'checks':n,'max_pair_formula_error':maximum,'max_actual_scalar_vs_full_residual_error':costmax,'max_actual_analytic_gradient_difference':gradmax,'milliseconds_per_16state_twofactor_score':(time.perf_counter()-start)*10,'H_only_graph_faces':len(graph['faces']),'H_only_graph_edges':len(graph['edges']),'GT_read':False,'finite_quadrature_not_full_graph_Bayes':True}


if __name__=='__main__':
    r=checks();save('research/sparse_material_evidence_checks_v1.json',r);print(r)
