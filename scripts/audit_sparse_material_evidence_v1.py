"""Reconstruct all shared-state costs and final objectives without task truth."""
from pathlib import Path
import pickle
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from inspection_decisions.sparse_material_evidence_v1 import material_states,PeriodModel,pair_costs,reduce_slots,log_weights,evidence_tensor,DT
from inspection_decisions.material_supported_continuous_alignment_v1 import world_pose


def dual_bound(C,tau):
    n=len(C);i,j,k=np.indices(C.shape);col=np.arange(C.size);A=coo_matrix((np.ones(3*C.size),(np.r_[i.ravel(),n+j.ravel(),2*n+k.ravel()],np.tile(col,3))),shape=(3*n,C.size)).tocsr();lp=linprog(C.ravel()/tau,A_eq=A,b_eq=np.ones(3*n),bounds=(0,None),method='highs')
    assert lp.success;y=lp.eqlin.marginals*tau;violation=max(0.,float((A.T@y-C.ravel()).max()));y-=violation/3+1e-15;assert float((A.T@y-C.ravel()).max())<=1e-14;return float(y.sum()),violation


def main():
    torch.set_num_threads(1);root=Path('results/sparse_material_evidence_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');manifest=read(root/'manifest.json');assert seal['complete'] and seal['scenes']==8;checks=0;maxcost=0.;maxfinal=0.;gaps=[];ess=[];fits=0;descent=0;constraintmax=0.;partialgraphs=[];mdl_roundoff=0.;mdl_bound=0.
    for name,h in seal['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;checks+=1
    methods=cfg['methods'];limits=manifest['motion_limits'];tau=cfg['evidence_temperature']
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;references=read(dest/'references.json');rows=read(dest/'predictions.json');assignments=read(dest/'assignments.json');starts=read(dest/'assignment_starts.json')
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3]
        with np.load(dest/'joint_costs.npz') as z:cost=z['cost'];ids=z['indices']
        assert cost.shape==(5,24,24,24) and np.isfinite(cost).all();checks+=1
        for i,reference in enumerate(references):
            graph,states=material_states(data[0][i],cfg,run['seed']*100+i);assert graph==reference['graph'];assert np.array_equal(states,reference['states']);checks+=2;partialgraphs.append({'scene':tag,'asset':i,'faces':len(graph['faces']),'termination':graph['termination_nodes'],'edges':len(graph['edges'])});D=torch.tensor(states,dtype=DT)
            for t in range(2):assert len(pools[i][t])==288 and {b['slot'] for b in pools[i][t]}==set(range(24)) and sum(b['catalogue'] for b in pools[i][t])==216;checks+=1
            with torch.inference_mode():H=PeriodModel(data[0][i],graph,cfg).evaluate(torch.zeros(6,dtype=DT),D,('ray','line'))
            with np.load(dest/f'asset_{i:02d}_evidence.npz') as z:
                E={mode:z[mode] for mode in ['ray','line']};U=z['used'];rgb=z['rgb'];W={mode:z['logw_'+mode] for mode in ['ray','line']}
            for mode in ['ray','line']:
                assert np.allclose(H[mode][0].numpy(),reference['H_'+mode+'_energy'],rtol=0,atol=1e-12);assert np.allclose(log_weights(H[mode][0].numpy(),tau),W[mode],rtol=0,atol=1e-12);assert abs(np.exp(W[mode]).sum()-1)<1e-12;checks+=3;ess.append(1/np.sum(np.exp(W[mode])**2))
            for mi,method in enumerate(methods):
                mode='line' if method=='joint_line' else 'ray';full=pair_costs(E[mode][0],E[mode][1],W[mode],tau,method,U[0],U[1])+rgb[0,:,None]+rgb[1,None,:];cc,ii=reduce_slots(full,pools[i]);error=float(np.max(abs(cc-cost[mi,i])));maxcost=max(maxcost,error);assert error<1e-12 and np.array_equal(ii,ids[mi,i]);checks+=2;v=rows[i]['methods'][method];j,k=v['slots'];a,b=v['candidate_indices'];assert [a,b]==ids[mi,i,j,k].tolist() and pools[i][0][a]['slot']==j and pools[i][1][b]['slot']==k;checks+=1;models=[PeriodModel(data[1][j],graph,cfg),PeriodModel(data[2][k],graph,cfg)];ww=torch.tensor(W[mode],dtype=DT)
                for stage,param in [('candidate','initial_parameters'),('final','final_parameters')]:
                    q=np.array(v[param]);used=method=='profile_mdl_ray'
                    with torch.inference_mode():ee1,uu1=models[0].evaluate(torch.tensor(q[:6],dtype=DT),D,(mode,),used)[mode];ee2,uu2=models[1].evaluate(torch.tensor(q[6:],dtype=DT),D,(mode,),used)[mode];value=float(evidence_tensor(ee1,ee2,ww,tau,method,uu1,uu2))
                    target=v['log']['initial_cost' if stage=='candidate' else 'final_cost'];error=abs(value-target);maxfinal=max(maxfinal,error);assert error<1e-11;checks+=1
                    if stage=='candidate':
                        delta=abs(value+rgb[0,a]+rgb[1,b]-cost[mi,i,j,k]);bound=1e-11
                        if method=='profile_mdl_ray':
                            # Frozen torch implementation promotes integer count *
                            # scalar to default float32. Two roundings: coefficient
                            # and product; min is 1-Lipschitz in per-state costs.
                            n=np.arange(graph['termination_nodes']+1);actual=(tau*np.log(2)*torch.tensor(n)).numpy();emulated=np.float32(tau*np.log(2))*n.astype(np.float32);assert actual.dtype==np.float32 and np.array_equal(actual,emulated);analytic=4*np.finfo(np.float32).eps*tau*np.log(2)*max(1,graph['termination_nodes']);assert np.max(abs(actual.astype(float)-tau*np.log(2)*n))<=analytic;bound+=analytic;mdl_roundoff=max(mdl_roundoff,delta);mdl_bound=max(mdl_bound,analytic);checks+=2
                        assert delta<=bound;checks+=1
                    for t,s in enumerate([j,k]):
                        assert np.allclose(world_pose(q[6*t:6*t+6],centers[s]),v['stages'][stage]['T'+str(t+1)],rtol=0,atol=1e-12);checks+=1
                        for off,limit in [(0,limits[0]),(3,limits[1])]:violation=float(np.linalg.norm(q[6*t+off:6*t+off+3])-limit);constraintmax=max(constraintmax,violation);assert violation<1e-9;checks+=1
                assert v['log']['final_cost']<=v['log']['initial_cost']+1e-12;checks+=1;fits+=1;descent+=v['log']['accepted']
        for mi,method in enumerate(methods):
            a,b=np.array(assignments[method]['slots']);assert sorted(a)==sorted(b)==list(range(24));value=float(cost[mi,np.arange(24),a,b].sum());assert abs(value-assignments[method]['log']['energy'])<1e-11;assert value<=min(float(cost[mi,np.arange(24),aa,bb].sum()) for aa,bb in starts)+1e-11;checks+=3
            for row in rows:assert row['methods'][method]['slots']==[int(a[row['asset']]),int(b[row['asset']])];checks+=1
            bound,viol=dual_bound(cost[mi],tau);assert bound<=value+1e-11;gaps.append({'scene':tag,'method':method,'feasible_energy':value,'numerically_feasible_LP_dual_bound':bound,'gap':value-bound,'dual_correction':viol,'certified_within_1e-9':value-bound<1e-9});checks+=1
        print('AUDITED',tag,flush=True)
    result={'passed':True,'auditor_sha256':sha(Path(__file__)),'checks':checks,'scenes':8,'assets':192,'fits':fits,'descent_fits':int(descent),'max_cube_cost_reconstruction_error':maxcost,'max_continuous_objective_reconstruction_error':maxfinal,'MDL_candidate_to_continuous_float32_roundoff_max':mdl_roundoff,'MDL_proven_IEEE_roundoff_bound_max':mdl_bound,'MDL_count_penalty_uses_float32_in_frozen_continuous_solver':True,'no_tolerance_changed_for_main_joint_model':True,'max_pose_domain_violation':constraintmax,'H_effective_states_min':float(min(ess)),'H_effective_states_median':float(np.median(ess)),'joint_assignment_certified_count':sum(g['certified_within_1e-9'] for g in gaps),'assignment_total':len(gaps),'max_assignment_dual_gap':max(g['gap'] for g in gaps),'all_288_candidates_present':True,'H_only_rebuilt_identically':True,'GT_read':False,'not_continuous_integral_convergence_certificate':True};save(root/'audit.json',result);save(root/'assignment_bound_audit.json',gaps);save(root/'reference_graph_audit.json',partialgraphs);print(result,flush=True)


if __name__=='__main__':main()
