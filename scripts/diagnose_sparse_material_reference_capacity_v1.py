"""POST-SEAL true-pose/ID intervention; never imported by an inference runner."""
from pathlib import Path
import pickle
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.diagnose_typed_material_incidence_graph_v2 import source_truth
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.sparse_material_evidence_v1 import PeriodModel,DT
from inspection_decisions.typed_material_incidence_graph_v1 import JointProblem


def main():
    torch.set_num_threads(1);root=Path('results/sparse_material_evidence_v1');assert read(root/'prediction_seal.json')['scenes']==8 and read(root/'audit.json')['passed'];cfg=read(root/'config.json');tau=cfg['evidence_temperature'];limits=read(root/'manifest.json')['motion_limits'];rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl');centers=inp['T0'][:,:3,3];geometry=geometry_for(run['seed'],24,run['geometry'])
        for i,r in enumerate(read(root/tag/'references.json')):
            G=r['graph'];D=torch.tensor(r['states'],dtype=DT);slots=[int(truth['slot1'][i]),int(truth['slot2'][i])];track=[data[0][i],data[1][slots[0]],data[2][slots[1]]];x=np.r_[pose_vector(truth['poses'][0,i],centers[slots[0]]),pose_vector(truth['poses'][1,i],centers[slots[1]])]
            with torch.inference_mode():E=[PeriodModel(d,G,cfg).evaluate(torch.tensor(x[t*6:t*6+6],dtype=DT),D)['ray'][0].numpy() for t,d in enumerate(track[1:])]
            w=np.array(r['logw_ray'])-(E[0]+E[1])/tau;w-=logsumexp(w);state=int(w.argmax());p=JointProblem(track,G,x,limits,cfg,'typed_ray');p.free_faces=False;p.face0=np.array(r['states'])[state];p.size=p.depth_start=12;rr,_,diag=p.evaluate(x,False);P=rr[:4*p.N].reshape(-1,4)/p.pweight[:,None];B=rr[4*p.N:].reshape(-1,5)/p.bweight[:,None]
            for t in range(3):
                s=i if t==0 else slots[t-1];T=inp['T0'][i] if t==0 else truth['poses'][t-1,i];labels=source_truth(track[t],T,centers[s],geometry[i]);available=0;covered=0
                for planes in labels:
                    if not planes:continue
                    available+=1;best=np.inf
                    for a,value in planes:
                        for F in G['faces']:
                            if F['axis']!=a:continue
                            for lo,hi in F['historical_intervals'] or [[F['offset']-cfg['face_mode_radius_m'],F['offset']+cfg['face_mode_radius_m']]]:best=min(best,max(lo-value,0,value-hi))
                    covered+=best<=.006
                pp=p.epoch==t;bb=p.be==t;rows.append({'scene':tag,'asset':i,'epoch':t,'pixels':int(pp.sum()),'capped_pixels_at_true_ID_and_pose':int(np.sum(np.linalg.norm(P[pp],axis=1)>=cfg['support_match_cap_m']-1e-10)),'rays':int(bb.sum()),'capped_rays_at_true_ID_and_pose':int(np.sum(np.linalg.norm(B[bb],axis=1)>=cfg['support_match_cap_m']-1e-10)),'true_source_pixels_available':available,'source_plane_covered_by_FULL_reference_intervals_within6mm':int(covered),'optimistic_ignores_finite_domains_and_noise_correlations':True})
        print('REFERENCE_CAPACITY',tag,flush=True)
    totals=[{k:sum(r[k] for r in rows if r['epoch']==t) for k in ['pixels','capped_pixels_at_true_ID_and_pose','rays','capped_rays_at_true_ID_and_pose','true_source_pixels_available','source_plane_covered_by_FULL_reference_intervals_within6mm']} for t in range(3)];report={'tasks':192,'epoch_totals':totals,'postseal_only':True,'GT_not_used_for_retuning_or_predictions':True,'not_task_success_or_oracle_algorithm_score':True,'script_sha256':sha(Path(__file__)),'truth_sha256':truth_hash};save(root/'reference_capacity_postseal.json',rows);save(root/'reference_capacity_summary.json',report);print(report,flush=True)


if __name__=='__main__':main()
