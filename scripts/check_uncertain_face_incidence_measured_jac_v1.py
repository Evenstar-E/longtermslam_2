"""Derivative probes on actual inference observations, without evaluation data."""
from pathlib import Path
import pickle
import numpy as np
from inspection_decisions.uncertain_face_incidence_graph_v1 import JointProblem,induce_graph
from scripts.run_rigid_identity_bounds_v1 import read,save
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector


def main():
    options=read('configs/uncertain_face_incidence_graph_v1.json');root=Path(options['output_root']);cfg={**read(Path(options['input_root'])/'config.json'),**options};tag='aligned_s270100_h1';old=Path(cfg['input_root']);base=Path(cfg['base_root']);rng=np.random.default_rng(9125)
    observed=root/tag/'observations.pkl'
    if not observed.exists():observed=Path('results/uncertain_face_incidence_graph_v1')/tag/'observations.pkl'
    with observed.open('rb') as f:data=pickle.load(f)
    with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
    center=inp['T0'][:,:3,3];source=read(old/tag/'predictions.json')[0];i=source['asset'];s1,s2=source['selected_slots'];track=[data[0][i],data[1][s1],data[2][s2]];initial=np.r_[pose_vector(np.array(source['initial_T1']),center[s1]),pose_vector(np.array(source['initial_T2']),center[s2])];graph=induce_graph([(np.zeros(6),track[0]),(initial[:6],track[1])],cfg);limits=read(Path(cfg['cached_model_root'])/'manifest.json')['motion_limits'];records=[]
    for method in cfg['methods']:
        problem=JointProblem(track,graph,initial,limits,cfg,method);p=problem.start();p+=rng.normal(0,1e-5,len(p));r,J,diag=problem.evaluate(p);columns=np.unique(np.r_[np.arange(min(problem.depth_start,problem.size)),rng.integers(0,problem.size,16)]).astype(int);maximum=0.;gradient_error=0.;switches=0
        for col in columns:
            hi=p.copy();lo=p.copy();hi[col]+=1e-7;lo[col]-=1e-7;rh,_,dh=problem.evaluate(hi,False);rl,_,dl=problem.evaluate(lo,False);numeric=(rh-rl)/2e-7;error=float(abs(numeric-J[:,col].toarray().ravel()).max());changed=any(not np.array_equal(dh[key],diag[key]) or not np.array_equal(dl[key],diag[key]) for key in ['pixel_face','edge','normal_id']);switches+=int(changed)
            if not changed:maximum=max(maximum,error)
            gradient_error=max(gradient_error,abs((rh@rh-rl@rl)/2e-7-float(2*(J[:,col].T@r)[0])))
        assert maximum<2e-5 and gradient_error<2e-5,(method,maximum,gradient_error);records.append({'method':method,'columns':len(columns),'maximum_residual_J_error_without_assignment_switch':maximum,'maximum_objective_gradient_error':gradient_error,'assignment_switch_probes':switches,'pixels':problem.N,'faces':problem.F,'edges':len(graph['edges'])})
    save(Path('research/uncertain_face_incidence_measured_jac_preflight_v1.json'),{'GT_read':False,'uses_previous_unsealed_observation_cache_only':True,'rows':records});print(records)


if __name__=='__main__':main()
