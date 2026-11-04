"""No-GT audit: full pools, one-to-one IDs, constraints and reconstructed energies."""
from pathlib import Path
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import catalog
from inspection_decisions.typed_material_incidence_graph_v1 import JointProblem
from inspection_decisions.typed_material_incidence_fast_v2 import period_arrays,period_geometry,graph_arrays,period_cost
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from scripts.run_typed_material_incidence_graph_v2 import mean_colors


def main():
    root=Path('results/typed_material_incidence_graph_v2');cfg=read(root/'config.json');manifest=read(root/'manifest.json');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8;checks=0;max_energy=0.;max_scalar=0.;nfits=0;anchored=0;accepted=0;global_deltas=[];stats=[];capacity_checks=[]
    for name,h in seal['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;checks+=1
    methods=cfg['methods'];families=['catalogue_plane','material_shared']
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['candidate_root'])/tag/'candidates.pkl').open('rb') as f:oldpool=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];rows=read(dest/'predictions.json');colors=mean_colors(Path(cfg['base_root']),tag,data);arrays=[[period_arrays(d) for d in tt] for tt in data]
        with np.load(dest/'global_costs.npz') as costs:
            for name in ['unary','catalogue_unary']:assert costs[name].shape==(2,4,2,24,24) and np.isfinite(costs[name]).all();checks+=1
            for i in range(24):
                for t in range(2):
                    pool=pools[i][t];assert len(pool)==288;assert sum(b['catalogue'] for b in pool)==216;checks+=2
                    for k,b in enumerate(pool[:108]):assert b['slot']==oldpool[i][t][k]['slot'] and np.array_equal(b['T'],oldpool[i][t][k]['T']) and b['cached_index']==k;checks+=1
                    for slot in range(24):
                        for C in catalog():
                            T=C.copy();T[:3,3]+=centers[slot];assert any(b['slot']==slot and b['catalogue'] and np.allclose(b['T'],T,rtol=0,atol=1e-12) for b in pool);checks+=1
            for fi,family in enumerate(families):
                rr=[r for r in rows if r['family']==family];assert [r['asset'] for r in rr]==list(range(24))
                for mi,m in enumerate(methods):
                    for stage,field,idx in [('pool','unary','indices'),('catalogue','catalogue_unary','catalogue_indices')]:
                        for t in range(2):
                            r,c=linear_sum_assignment(costs[field][fi,mi,t]);slots=[x['methods'][m]['stages'][stage]['slots'][t] for x in rr];assert slots==c.tolist() and len(set(slots))==24;checks+=1
                            for i,s in enumerate(slots):
                                choice=rr[i]['methods'][m]['selection'][stage];k=choice['indices'][t];b=pools[i][t][k];assert k==costs[idx][fi,mi,t,i,s] and b['slot']==s;assert stage!='catalogue' or b['catalogue'];assert np.array_equal(b['T'],rr[i]['methods'][m]['stages'][stage]['T'+str(t+1)]);checks+=3
        for row in rows:
            i=row['asset'];base=row['graphs']['clipped'];typed=row['graphs']['typed'];assert typed['faces'][:len(base['faces'])]==base['faces'] and typed['edges'][:len(base['edges'])]==base['edges'];checks+=1;stats.append({'scene':tag,'family':row['family'],'asset':i,'source_nodes':len(base['faces']),'termination_nodes':typed['termination_nodes'],'edges_before':len(base['edges']),'edges_after':len(typed['edges'])})
            # Enlarging the dictionary must allow a feasible extension of the old
            # graph. Lower energy alone is therefore NOT evidence of better geometry.
            ss=row['initial_slots'];problem=JointProblem([data[0][i],data[1][ss[0]],data[2][ss[1]]],typed,row['initial_parameters'],manifest['motion_limits'],cfg,'typed_ray');old=np.array(row['initial_fits']['clipped_ray']['parameters']);q=problem.start();q[:len(old)]=old;q=problem.project(q);r,_,_=problem.evaluate(q,False);feasible=float(r@r);assert feasible<=row['initial_fits']['clipped_ray']['log']['final_cost']+1e-11;checks+=1;capacity_checks.append({'scene':tag,'asset':i,'family':row['family'],'clipped_final_cost':row['initial_fits']['clipped_ray']['log']['final_cost'],'typed_feasible_extension_cost':feasible,'typed_own_final_cost':row['initial_fits']['typed_ray']['log']['final_cost'],'typed_own_worse_than_feasible_extension':bool(row['initial_fits']['typed_ray']['log']['final_cost']>feasible+1e-10)})
            for m in methods:
                graph=base if m=='clipped_ray' else typed;v=row['methods'][m];original_slots=row['initial_slots'];slots=v['stages']['final']['slots'];assert slots==v['stages']['pool']['slots'];checks+=1;energies=[]
                for final in [False,True]:
                    ss=slots if final else original_slots;x=v['initial_parameters'] if final else row['initial_parameters'];track=[data[0][i],data[1][ss[0]],data[2][ss[1]]];p=JointProblem(track,graph,x,manifest['motion_limits'],cfg,m);q=np.array(v['final_parameters'] if final else row['initial_fits'][m]['parameters']);log=v['log'] if final else row['initial_fits'][m]['log'];assert np.allclose(p.project(q),q,rtol=0,atol=1e-12);r,_,_=p.evaluate(q,False);energy=float(r@r);error=abs(energy-log['final_cost']);max_energy=max(max_energy,error);assert error<1e-11;checks+=2;nfits+=1
                    rawsum=sum(period_cost(period_geometry(arrays[t][i if t==0 else ss[t-1]],np.zeros(6) if t==0 else q[6*(t-1):6*t],cfg),graph_arrays(graph,q[12:]),cfg,m=='line_joint') for t in range(3));error=abs(rawsum-energy);max_scalar=max(max_scalar,error);assert error<1e-11;checks+=1
                    before=log['initial_cost']
                    for step in log['accepted_steps']:assert abs(before-step['before'])<1e-11 and step['after']<before;before=step['after'];checks+=1;accepted+=1
                    assert abs(before-energy)<1e-11;checks+=1
                    if m!='ray_no_history':
                        for j,F in enumerate(graph['faces']):
                            if F['historical_intervals']:assert any(lo-1e-12<=q[12+j]<=hi+1e-12 for lo,hi in F['historical_intervals']);checks+=1;anchored+=1
                    for t in range(2):assert np.allclose(world_pose(q[6*t:6*t+6],centers[ss[t]]),v['stages']['final' if final else 'conditional']['T'+str(t+1)],rtol=0,atol=1e-12);checks+=1
                    energies.append(energy)
                # Final fitting must descend from the actual pooled identity/pose and inherited shared material.
                q=np.r_[v['initial_parameters'],row['initial_fits'][m]['parameters'][12:]];p=JointProblem([data[0][i],data[1][slots[0]],data[2][slots[1]]],graph,v['initial_parameters'],manifest['motion_limits'],cfg,m);r,_,_=p.evaluate(p.project(q),False);assert abs(float(r@r)-v['log']['initial_cost'])<1e-11;checks+=1
                oldcolor=sum(cfg['support_match_cap_m']**2/3*float(np.mean(((colors[0,i]-colors[t+1,original_slots[t]])/255)**2)) for t in range(2));newcolor=sum(cfg['support_match_cap_m']**2/3*float(np.mean(((colors[0,i]-colors[t+1,slots[t]])/255)**2)) for t in range(2));global_deltas.append({'scene':tag,'family':row['family'],'method':m,'asset':i,'delta_pool_energy':v['log']['initial_cost']+newcolor-energies[0]-oldcolor})
        print('AUDITED',tag,flush=True)
    # One-to-one matching is globally improving, not necessarily improving every individual row.
    max_delta=-np.inf
    for scene in {r['scene'] for r in global_deltas}:
        for family in families:
            for m in methods:
                delta=sum(r['delta_pool_energy'] for r in global_deltas if (r['scene'],r['family'],r['method'])==(scene,family,m));max_delta=max(max_delta,delta);assert delta<1e-10;checks+=1
    report={'passed':True,'auditor_sha256':sha(Path(__file__)),'checks':checks,'scenes':8,'fits_reconstructed':nfits,'max_energy_reconstruction_error':max_energy,'max_fast_vs_full_residual_energy_error':max_scalar,'max_global_assignment_energy_delta':max_delta,'historical_interval_membership_checks':anchored,'accepted_descent_steps':accepted,'all108_original_candidates_retained':True,'all24_slots_times9_catalogue_present':True,'same288_candidates_per_asset_period':True,'all_global_assignments_one_to_one':True,'typed_own_worse_than_clipped_feasible_extension':sum(r['typed_own_worse_than_feasible_extension'] for r in capacity_checks),'GT_read':False,'same_observations_not_equal_FLOPs':True};save(root/'audit.json',report);save(root/'graph_structure_audit.json',stats);save(root/'global_energy_audit.json',global_deltas);save(root/'capacity_dominance_audit.json',capacity_checks);print(report,flush=True)


if __name__=='__main__':main()
