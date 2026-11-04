"""Profiled supports, observable modes and GT-based face-association diagnostics."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES,historical_offsets,material_residual,pose_vector,right_jacobian,skew


def main():
    root=Path('results/material_supported_continuous_alignment_v2');cfg=read(root/'config.json');base=Path(cfg['base_root'])
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    supports=[];true_features=[];opportunities=[];counts={'icp_paths':0,'icp_no_accepted_step':0,'icp_zero_rank_final_level':0,'icp_accepted_steps':0,'joint_paths':0,'joint_no_accepted_step':0}
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (dest/'features.pkl').open('rb') as f:features=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (base/tag/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(dest/'predictions.json');slots=inp['T0'][:,:3,3]
        for log in read(dest/'bounded_icp_logs.json'):
            steps=sum(len(v['accepted_steps']) for v in log['levels']);counts['icp_paths']+=1;counts['icp_no_accepted_step']+=steps==0;counts['icp_accepted_steps']+=steps
            counts['icp_zero_rank_final_level']+=bool(log['levels'] and log['levels'][-1]['rank']==0)
        for log in read(dest/'joint_logs.json'):counts['joint_paths']+=1;counts['joint_no_accepted_step']+=not log['accepted_steps']
        for name in ['material_independent','material_shared']:
            selection=pred[name]['selection']
            if not selection:continue
            for i,k,l in selection:
                ref=historical_offsets(features[0][i]);chosen=[candidates[i][0][k],candidates[i][1][l]];ff=[features[t+1][b['slot']] for t,b in enumerate(chosen)];centers=[slots[b['slot']] for b in chosen]
                x=np.concatenate([pose_vector(b['T'],c) for b,c in zip(chosen,centers)]);r,J=material_residual(x,ff,ref,cfg,name=='material_shared')
                _,singular,V=np.linalg.svd(J,full_matrices=True);rank=int((singular>1e-7).sum());null=V[rank:].T;B=np.zeros((3,12));a=inp['anchors'][i];b_est={};visible={}
                for t in [0,1]:
                    R=chosen[t]['T'][:3,:3];B[:,6*t:6*t+3]=(-1 if t==0 else 1)*(-R@skew(a)@right_jacobian(x[6*t:6*t+3]));B[:,6*t+3:6*t+6]=(-1 if t==0 else 1)*np.eye(3)
                    for f in ff[t]:
                        axis=f['axis'];q=AXES[axis]@R.T@(f['centroid']-x[6*t+3:6*t+6]);b_est.setdefault(axis,[]).append(float(q));visible.setdefault(axis,set()).add(t)
                sensitivity=float(np.linalg.norm(B@null,2)) if null.size else 0.
                born=[axis for axis in b_est if axis not in ref];shared_born=[axis for axis in born if len(visible[axis])==2]
                supports.append({'scene':tag,'method':name,'asset':i,'data_jacobian_rank':rank,'local_anchor_nullspace_sensitivity':sensitivity,'missing_data_mode_can_change_anchor':sensitivity>1e-6,
                    'profiled_born_offsets':{str(axis):float(np.mean(b_est[axis])) for axis in born},'current_born_shared_axes':shared_born,
                    'interpretation':'Derived from sealed predictions, no GT used in these readouts. Unconstrained data-nullspace diagnostic ignores active ball boundaries; not a calibrated physical certificate.'})
        for i in range(24):
            ref=historical_offsets(features[0][i]);current=[features[t+1][int(truth[f'slot{t+1}'][i])] for t in [0,1]];present=[set(f['axis'] for f in ff) for ff in current];born=sorted((present[0]&present[1])-set(ref));qs={}
            for t,ff in enumerate(current):
                T=truth['poses'][t,i];slot=int(truth[f'slot{t+1}'][i]);u=T[:3,3]-slots[slot]
                for f in ff:
                    axis=f['axis'];q=float(AXES[axis]@T[:3,:3].T@(f['centroid']-u));angle=float(np.degrees(np.arccos(np.clip((T[:3,:3]@AXES[axis])@f['normal'],-1,1))))
                    qs.setdefault((t,axis),[]).append(q);true_features.append({'scene':tag,'asset':i,'epoch':t,'axis':axis,'view':f['view'],'normal_error_deg':angle,
                        'history_reference_available':axis in ref,'offset_minus_history_m':q-ref[axis] if axis in ref else None,'measured_fit_rms_m':f['rms_m']})
            opportunities.append({'scene':tag,'asset':i,'true_id_born_shared_axes':born,'true_pose_cross_epoch_offset_gaps_m':{str(axis):float(np.mean(qs[(0,axis)])-np.mean(qs[(1,axis)])) for axis in born}})
    selected_shared=[s for s in supports if s['method']=='material_shared'];summary={**counts,
        'selected_shared_assets_with_born_track':sum(bool(s['current_born_shared_axes']) for s in selected_shared),
        'selected_shared_born_tracks':sum(len(s['current_born_shared_axes']) for s in selected_shared),
        'selected_shared_anchor_sensitive_data_nullspace':sum(s['missing_data_mode_can_change_anchor'] for s in selected_shared),
        'true_id_assets_with_born_shared_track':sum(bool(o['true_id_born_shared_axes']) for o in opportunities),
        'true_id_born_shared_tracks':sum(len(o['true_id_born_shared_axes']) for o in opportunities),
        'observed_face_features_under_true_ids':len(true_features),
        'historical_reference_offset_disagreement_gt_12mm':sum(abs(f['offset_minus_history_m'])>.012 for f in true_features if f['history_reference_available']),
        'true_born_track_offset_gap_gt_12mm':sum(abs(gap)>.012 for o in opportunities for gap in o['true_pose_cross_epoch_offset_gaps_m'].values()),
        'normal_error_gt_10deg':sum(f['normal_error_deg']>10 for f in true_features)}
    save(root/'mechanism_diagnosis.json',{'summary':summary,'postseal':True,'predictions_changed':False,'GT_used_only_in_true_id_feature_checks':True,'support_readouts':supports,'true_id_opportunities':opportunities,'true_pose_feature_consistency':true_features,'source_sha256':sha(Path(__file__))})
    print(summary,flush=True)


if __name__=='__main__':main()
