"""Verify same-energy Schur repair and all common endpoint choices, no GT."""
from pathlib import Path
import pickle
import numpy as np
from inspection_decisions.uncertain_face_incidence_graph_v1 import JointProblem
from inspection_decisions.material_supported_continuous_alignment_v1 import world_pose
from scripts.run_uncertain_face_incidence_graph_v2 import choices_for
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/uncertain_face_incidence_graph_v2');cfg=read(root/'config.json');parent=Path(cfg['parent_root']);assert read(root/'prediction_seal.json')['complete'];manifest=read(root/'manifest.json');checks=0;maximum=0.;rotations={m:0 for m in cfg['methods']};sources={m:{} for m in cfg['methods']};steps={m:0 for m in cfg['methods']};bounds={m:0 for m in cfg['methods']};dominance_failures_before=0;dominance_failures_after=0;latent_variants=[]
    for rr in [root,parent]:
        for name,h in read(rr/'prediction_seal.json')['sha256'].items():assert sha(rr/name)==h;checks+=1
        for field in ['sources','inputs']:
            for name,h in read(rr/'manifest.json')[field].items():assert sha(name)==h;checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];prior=read(parent/tag/'predictions.json');rows=read(root/tag/'predictions.json');assert len(rows)==len(prior)==48
        with (parent/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3]
        for row,old in zip(rows,prior):
            assert (row['asset'],row['family'],row['selected_slots'],row['graph'])==(old['asset'],old['family'],old['selected_slots'],old['graph']);checks+=1;i=row['asset'];s1,s2=row['selected_slots'];track=[data[0][i],data[1][s1],data[2][s2]];problems={m:JointProblem(track,row['graph'],row['initial_parameters'],manifest['motion_limits'],cfg,m) for m in cfg['methods']};pool=[('original',problems['normal_joint'],problems['normal_joint'].start())]+[(m,problems[m],np.array(old['parameters'][m])) for m in cfg['methods']]+[('schur_'+m,problems[m],np.array(row['fit_parameters'][m])) for m in cfg['methods']];assert len(pool)==9;checks+=1
            dominance_failures_before+=int(old['logs']['bounded_joint']['final_cost']>old['logs']['normal_joint']['final_cost']+1e-10);dominance_failures_after+=int(row['common_pool_selection']['bounded_joint']['energy']>row['common_pool_selection']['normal_joint']['energy']+1e-10)
            assert row['common_pool_selection']['bounded_joint']['energy']<=row['common_pool_selection']['normal_joint']['energy']+1e-10;checks+=1
            for method,p in problems.items():
                q=np.array(row['parameters'][method]);assert np.allclose(p.project(q),q,rtol=0,atol=1e-12);r,_,_=p.evaluate(q,False);err=abs(float(r@r)-row['common_pool_selection'][method]['energy']);maximum=max(maximum,err);assert err<1e-11;checks+=2
                choices=choices_for(p,pool);value,name,best=min(choices,key=lambda x:x[0]);assert name==row['common_pool_selection'][method]['source'];assert np.allclose(best,q,rtol=0,atol=1e-13);assert row['common_pool_selection'][method]['pool_size']==9 and row['common_pool_selection'][method]['latent_depth_variants']==len(choices);checks+=3;latent_variants.append(len(choices))
                own=np.array(row['fit_parameters'][method]);rr,_,_=p.evaluate(own,False);log=row['logs'][method];assert abs(float(rr@rr)-log['final_cost'])<1e-11;assert value<=log['final_cost']+1e-11;checks+=2;cost=log['initial_cost']
                for step in log['accepted_steps']:assert abs(step['before']-cost)<1e-11 and step['after']<cost;cost=step['after'];checks+=1
                assert abs(cost-log['final_cost'])<1e-11;steps[method]+=len(log['accepted_steps']);checks+=1;sources[method][name]=sources[method].get(name,0)+1;bounds[method]+=log['depths_at_bounds']
                for t,slot in enumerate([s1,s2]):assert np.allclose(world_pose(q[6*t:6*t+6],centers[slot]),row['predictions'][method]['T'+str(t+1)],atol=1e-13,rtol=0);checks+=1
                initial=np.array(row['initial_parameters']);rotations[method]+=int(np.linalg.norm(q[:3]-initial[:3])>1e-7 or np.linalg.norm(q[6:9]-initial[6:9])>1e-7)
    report={'checks':checks,'conditional_tracks':384,'fits':1536,'all9_common_pose_endpoints_checked':True,'max_energy_reconstruction_error':maximum,'old_bounded_energy_worse_than_feasible_point_count':dominance_failures_before,'new_bounded_energy_worse_than_point_count':dominance_failures_after,'pooled_predictions_actually_update_rotation':rotations,'accepted_Schur_steps':steps,'fit_depth_variables_at_bounds':bounds,'selected_sources':sources,'max_latent_variants_per9_poses':max(latent_variants),'GT_read':False,'same_observations_not_equal_FLOPs_or_latent_capacity':True};save(root/'audit.json',report);print(report)


if __name__=='__main__':main()
