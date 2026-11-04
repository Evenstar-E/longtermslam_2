"""Measure real coupling and finite-state degeneracy; no truth geometry access."""
from pathlib import Path
import pickle
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from inspection_decisions.sparse_material_evidence_v1 import PeriodModel,DT
from inspection_decisions.typed_material_incidence_graph_v1 import JointProblem


def main():
    torch.set_num_threads(1);root=Path('results/sparse_material_evidence_v1');assert read(root/'prediction_seal.json')['scenes']==8 and read(root/'audit.json')['passed'];cfg=read(root/'config.json');tau=cfg['evidence_temperature'];rows=[];coupling=[];sat=[];limits=read(root/'manifest.json')['motion_limits']
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;refs=read(dest/'references.json');pred=read(dest/'predictions.json')
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with np.load(dest/'joint_costs.npz') as z:C=z['cost']
        for i,r in enumerate(refs):
            item={'scene':tag,'asset':i,'H_effective_states':r['H_effective_states_ray'],'joint_vs_independent_different_slots':pred[i]['methods']['joint_ray']['slots']!=pred[i]['methods']['independent_ray']['slots']};D=torch.tensor(r['states'],dtype=DT);W=np.array(r['logw_ray']);v=pred[i]['methods']['joint_ray'];j,k=v['slots'];x=np.array(v['final_parameters']);g=r['graph']
            with torch.inference_mode():E=[PeriodModel(data[t+1][slot],g,cfg).evaluate(torch.tensor(x[t*6:t*6+6],dtype=DT),D)['ray'][0].numpy() for t,slot in enumerate([j,k])]
            logpost=W-(E[0]+E[1])/tau;logpost-=logsumexp(logpost);item['final_effective_states']=float(1/np.sum(np.exp(logpost)**2));item['MAP_state']=int(logpost.argmax());L1=np.exp(-(E[0]-E[0].min())/tau);L2=np.exp(-(E[1]-E[1].min())/tau);weight=np.exp(W);item['selected_pair_shared_correction_over_tau']=float(-np.log(np.sum(weight*L1*L2)/(np.sum(weight*L1)*np.sum(weight*L2))));item['state_energy_spans_over_tau']=[float(np.ptp(e)/tau) for e in E]
            for method in ['joint_ray','independent_ray','profile_ray','profile_mdl_ray','joint_line']:
                a=C[cfg['methods'].index(method),i];centered=a-a.mean(0,keepdims=True)-a.mean(1,keepdims=True)+a.mean();q=pred[i]['methods'][method];initial=np.array(q['initial_parameters']);final=np.array(q['final_parameters']);coupling.append({'scene':tag,'asset':i,'method':method,'max_nonseparable_cost_over_tau':float(abs(centered).max()/tau),'rms_nonseparable_cost_over_tau':float(np.sqrt((centered**2).mean())/tau),'max_rotation_change_rad':float(max(np.linalg.norm(final[:3]-initial[:3]),np.linalg.norm(final[6:9]-initial[6:9]))),'max_translation_change_m':float(max(np.linalg.norm(final[3:6]-initial[3:6]),np.linalg.norm(final[9:12]-initial[9:12]))),'objective_drop':q['log']['initial_cost']-q['log']['final_cost']})
            # MAP-state cap rates diagnose an incomplete material dictionary; not
            # true material correspondence or a causal oracle intervention.
            p=JointProblem([data[0][i],data[1][j],data[2][k]],g,x,limits,cfg,'typed_ray');p.free_faces=False;p.face0=np.array(r['states'])[item['MAP_state']];p.size=p.depth_start=12;res,_,diag=p.evaluate(x,False);P=res[:4*p.N].reshape(-1,4)/p.pweight[:,None];B=res[4*p.N:].reshape(-1,5)/p.bweight[:,None]
            for t in range(3):pp=p.epoch==t;bb=p.be==t;sat.append({'scene':tag,'asset':i,'epoch':t,'pixels':int(pp.sum()),'capped_pixels':int(np.sum(np.linalg.norm(P[pp],axis=1)>=cfg['support_match_cap_m']-1e-10)),'rays':int(bb.sum()),'capped_rays':int(np.sum(np.linalg.norm(B[bb],axis=1)>=cfg['support_match_cap_m']-1e-10))})
            edge_sets=[set(diag['edge'][p.be==t].tolist())-{-1} for t in [1,2]];item['MAP_C1_edges']=len(edge_sets[0]);item['MAP_C2_edges']=len(edge_sets[1]);item['MAP_shared_current_edge_instances']=len(edge_sets[0]&edge_sets[1]);rows.append(item)
        print('DIAGNOSED',tag,flush=True)
    summary={'assets':len(rows),'H_effective_states_median':float(np.median([r['H_effective_states'] for r in rows])),'final_effective_states_median':float(np.median([r['final_effective_states'] for r in rows])),'final_effective_states_below2':sum(r['final_effective_states']<2 for r in rows),'joint_vs_independent_slot_changes':sum(r['joint_vs_independent_different_slots'] for r in rows),'MAP_no_shared_current_edge':sum(r['MAP_shared_current_edge_instances']==0 for r in rows),'selected_pair_shared_correction_over_tau_median':float(np.median([r['selected_pair_shared_correction_over_tau'] for r in rows])),'nonseparable_max_over_tau_by_method':{m:float(max(r['max_nonseparable_cost_over_tau'] for r in coupling if r['method']==m)) for m in cfg['methods']},'translation_updates_over1mm':{m:sum(r['max_translation_change_m']>.001 for r in coupling if r['method']==m) for m in cfg['methods']},'MAP_capped_by_epoch':[{k:sum(r[k] for r in sat if r['epoch']==t) for k in ['pixels','capped_pixels','rays','capped_rays']} for t in range(3)],'GT_read':False,'diagnostic_not_continuous_quadrature_convergence':True,'script_sha256':sha(Path(__file__))};save(root/'material_state_diagnostic.json',rows);save(root/'coupling_diagnostic.json',coupling);save(root/'MAP_cap_diagnostic.json',sat);save(root/'diagnostic_summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
