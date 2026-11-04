"""Evaluator-only proposal provenance check, not an identity prediction method."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem

def main():
    root=Path('data/livingscenes_native_masks_v1');public=read(root/'public.json');truth=read('data/real_cross_session_v1/prepared_v2/evaluation/identity.json');scores=read('results/candidate_complete_v1/scores.json');rows=[]
    for scene in ['scene_a','scene_c']:
        pred=read(Path('results/candidate_complete_v1')/scene/'predictions.json');proposals=read(Path('results/candidate_complete_v1')/scene/'pair_proposals.json');data={}
        for period in ['0','1']:
            data[period]=[]
            for token in pred['history' if period=='0' else 'current']:
                e=next(r for r in public[scene][period] if r['token']==token)
                with np.load(root/e['path']) as z:item=dict(e,xyz=z['xyz'],rgb=z['rgb'])
                with np.load(Path('results/livingscenes_native_masks_v1/embeddings')/scene/period/(token+'.npz')) as z:item['descriptor']=z['z_inv'].reshape(-1)
                data[period].append(item)
        H,C=data['0'],data['1'];h=np.array([v['descriptor'] for v in H]);c=np.array([v['descriptor'] for v in C]);h/=np.linalg.norm(h,axis=1,keepdims=True);c/=np.linalg.norm(c,axis=1,keepdims=True)
        p=CachedJointProblem(H,C,np.clip((1-c@h.T)/2,0,1),read('configs/rigid_identity_bounds_v1.json'))
        for d in scores['candidate_diagnosis']:
            if d['scene']!=scene or not d['true_id_present_complete'] or d['true_id_present_gated']:continue
            j=pred['current'].index(d['token']);truecols=[i for i,t in enumerate(pred['history']) if truth[scene]['0'][t]==d['dominant_truth_id']]
            for i in truecols:
                trials=[]
                for seed in range(len(C)):
                    T=np.array(proposals['poses'][f'{seed},{i}']);dist=p.tree[i].query((C[j]['xyz']-T[:3,3])@T[:3,:3],workers=1)[0];geometry=.6*np.mean(np.minimum(dist/.075,1)**2);motion=.2*min(np.linalg.norm(T[:3,:3]@p.anchor[i]+T[:3,3]-p.anchor[i]),1.)
                    trials.append(dict(seed_current=pred['current'][seed],energy=float(geometry+p.appearance[j,i]+motion),geometry=float(geometry),appearance=float(p.appearance[j,i]),motion=float(motion)))
                T=np.array(pred['methods']['complete_common_rigid']['poses'][i]);rows.append(dict(scene=scene,current_token=d['token'],oracle_true_history=pred['history'][i],pair_own_seed_cost=trials[j],best_other_current_seed=min(trials,key=lambda r:r['energy']),cost_at_common_selected_pose=p.pair_cost(j,i,T),all_seed_costs=trials))
    save('results/candidate_complete_v1/oracle_pose_seed_diagnosis.json',dict(rows=rows,scope='GT selects which archive to diagnose AFTER sealing all predictions; no independent motion truth, no GT-derived inference candidate. All tested poses pre-existing label-blind pair proposals. All-identity inclusion is not complete continuous pose coverage.'))
    print(rows)

if __name__=='__main__':main()
