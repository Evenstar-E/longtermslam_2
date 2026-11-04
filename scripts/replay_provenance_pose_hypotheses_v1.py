"""Post-seal chosen-branch replay: relate surrogate improvements to truth changes.

No new branch is selected with truth. Replays exactly the already sealed decisions.
Event counts may revisit a voxel; final unique changes are in the primary scorer.
"""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha,load
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem


def main():
    root=Path('results/provenance_pose_hypotheses_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    rows=[];summaries=[]
    for scene in cfg['scenes']:
        H,O,raw,initial,_=load(scene,cfg);ids=np.array([h['uid'] for h in H]);eligible={h['uid'] for h in H if h['class'] not in [0,1,2,22] and h['raw_points']>=128}
        disk=np.load(root/scene/'pool.npz');pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
        truths=[];allows=[];persistent=[]
        for t in [0,1]:
            base=Path('data/real_cross_session_v1/rescan/raw') if t==0 else Path('data/rescan_third_capture_v1/evaluation/raw')
            truth=read_ply(base/scene/'gt_segmentation'/f'{scene}_{t+1}.ply')['instance_idx'][O[t]['raw_index']];truths.append(truth)
            eq={};file=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_{t+1}.txt'
            if file.exists():
                for line in file.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        left,right=line.split('|');eq[int(left)]={int(v) for v in right.split()}
            allows.append(np.array([[int(uid) in (eq.get(int(tid),{int(tid)})&eligible) for uid in ids] for tid in truth],bool))
            persistent.append(np.isin(truth,list(eligible)))
        def correct(t,z):
            ok=z>=0;answer=np.zeros(len(z),bool);answer[ok]=allows[t][np.flatnonzero(ok),z[ok]];return answer
        for method in cfg['methods']:
            state=np.load(root/scene/method/'state.npz');native=[state['initial_labels1'],state['initial_labels2']]
            p=Problem(O,H,pools,native,cfg,method);trace=read(root/scene/method/'trace.json');events=[]
            for step,row in enumerate(trace):
                if not row['accepted']:continue
                t,i=row['period']-1,row['asset_index'];k=row['new_k'];T=pools[t][i][k]
                inc=p.surface(1-t,i) if t==1 or p.bidir else None
                col=p.column(t,i,T,inc);z,own,region=p.label_proposal(t,i,col)
                back,backcols,_=p.backward(t,i,T,z) if p.bidir else (0.,{},[])
                assert abs(own-row['own_delta'])<2e-7 and abs(back-row['outgoing_delta'])<2e-7
                before=correct(t,p.z[t]);after=correct(t,z);mask=persistent[t];changed=z!=p.z[t]
                fixed=changed&mask&~before&after;damaged=changed&mask&before&~after
                item={'scene':scene,'method':method,'trace_step':step,'sweep':row['sweep'],'period':t+1,'asset_uid':H[i]['uid'],'selected_k':int(k),
                      'own_delta':own,'outgoing_delta':back,'joint_delta':own+back,'changed_voxels':int(changed.sum()),
                      'equivalent_fixed_events':int(fixed.sum()),'equivalent_damaged_events':int(damaged.sum()),
                      'fixed_truth_distribution':{str(int(v)):int(c) for v,c in zip(*np.unique(truths[t][fixed],return_counts=True))},
                      'damaged_truth_distribution':{str(int(v)):int(c) for v,c in zip(*np.unique(truths[t][damaged],return_counts=True))}}
                events.append(item);rows.append(item)
                p.pose[t][i]=T.copy();p.k[t,i]=k;p.z[t]=z;p.C[t][:,i]=col
                for j,cc in backcols.items():p.C[1-t][:,j]=cc
                assert abs(p.energy()-row['energy_after'])<2e-7
            assert np.array_equal(p.k,state['k']) and np.array_equal(p.z[0],state['labels1']) and np.array_equal(p.z[1],state['labels2'])
            summaries.append({'scene':scene,'method':method,'accepted_events':len(events),'exact_final_state_reproduced':True,
                'joint_decrease_but_net_equivalent_damage_events':sum(e['joint_delta']<0 and e['equivalent_damaged_events']>e['equivalent_fixed_events'] for e in events),
                'period1_fixed_events':sum(e['equivalent_fixed_events'] for e in events if e['period']==1),
                'period1_damaged_events':sum(e['equivalent_damaged_events'] for e in events if e['period']==1)})
            print('REPLAYED',scene,method,len(events),flush=True)
    save(root/'accepted_branch_replay.json',{'summaries':summaries,'rows':rows,'source_sha':sha(__file__),
        'scope':'Evaluator-only replay of sealed choices, not oracle intervention. Counts are update events, may count the same voxel multiple times. Allowed equivalences used, only initial persistent assets. No new predictions.'})


if __name__=='__main__':main()
