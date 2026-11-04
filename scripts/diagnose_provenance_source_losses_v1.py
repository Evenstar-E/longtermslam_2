"""Disentangle final unsupported/null ties from known-ID source corruption.

Final-state conditional alternatives are diagnostic, not causal attribution of
every past update. Additionally replay the first scene_c asset4 source update.
"""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha,load
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem
from inspection_decisions.reversible_surface_v2 import costs


def main():
    root=Path('results/provenance_pose_hypotheses_v1');seal=read(root/'prediction_seal.json');assert seal['complete']
    for path,h in seal['sha256'].items():assert sha(root/path)==h
    cfg=read(root/'config.json');rows=[];event=None
    for scene in cfg['scenes']:
        H,O,_,_,_=load(scene,cfg);ids=np.array([h['uid'] for h in H]);eligible={h['uid'] for h in H if h['class'] not in [0,1,2,22] and h['raw_points']>=128}
        disk=np.load(root/scene/'pool.npz');pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
        truth=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply')['instance_idx'][O[0]['raw_index']]
        eq={};file=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if file.exists():
            for line in file.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    a,b=line.split('|');eq[int(a)]={int(x) for x in b.split()}
        def correct(z):
            pred=np.where(z>=0,ids[np.maximum(z,0)],-1)
            return np.array([int(u) in (eq.get(int(v),{int(v)})&eligible) for u,v in zip(pred,truth)])
        for method in ['pool_detached','pool_coupled']:
            state=np.load(root/scene/method/'state.npz');p=Problem(O,H,pools,[state['initial_labels1'],state['initial_labels2']],cfg,method)
            p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.z=[state['labels1'].copy(),state['labels2'].copy()]
            old=state['initial_labels1'];new=p.z[0];damage=np.isin(truth,list(eligible))&correct(old)&~correct(new)
            oldcost=np.full(len(old),np.nan)
            for i in np.unique(old[damage]):
                assert i>=0;idx=np.flatnonzero(damage&(old==i));obs={k:O[0][k][idx] for k in ['xyz','rgb','class']}
                oldcost[idx]=costs(obs,[p.pose[0][i]],[H[i]],[p.surface(1,i)],cfg)[0][:,0]
            null=damage&(new<0)
            rows.append({'scene':scene,'method':method,'equivalent_damaged_voxels':int(damage.sum()),
                         'now_unknown':int(null.sum()),'now_wrong_known_id':int(np.sum(damage&(new>=0))),
                         'unknown_and_original_identity_also_null_cost':int(np.sum(null&(oldcost>=p.null))),
                         'unknown_but_original_identity_supported':int(np.sum(null&(oldcost<p.null)))})
        if scene=='scene_c':
            state=np.load(root/scene/'pool_coupled/state.npz');p=Problem(O,H,pools,[state['initial_labels1'],state['initial_labels2']],cfg,'pool_coupled')
            for step,row in enumerate(read(root/scene/'pool_coupled/trace.json')):
                if not row['accepted']:continue
                t,i=row['period']-1,row['asset_index'];T=pools[t][i][row['new_k']]
                col=p.column(t,i,T,p.surface(1-t,i));z,own,_=p.label_proposal(t,i,col);back,bc,_=p.backward(t,i,T,z)
                assert abs(own-row['own_delta'])<2e-7 and abs(back-row['outgoing_delta'])<2e-7
                if t==0 and row['asset_uid']==4:
                    damage=np.isin(truth,list(eligible))&correct(p.z[0])&~correct(z)
                    C=p.C[0].copy();C[:,i]=col;before=p.assigned(0);after_old=p.assigned(0,p.z[0],C);after=p.assigned(0,z,C)
                    pure=damage&(z<0)&(before>=p.null)&(after_old>=p.null)
                    event={'scene':scene,'trace_step':step,'asset_uid':4,'damage':int(damage.sum()),'to_unknown':int(np.sum(damage&(z<0))),
                           'pure_local_null_tie':int(pure.sum()),'damaged_points_own_energy_delta':float(np.dot(p.weights[0][damage],after[damage]-before[damage])),
                           'own_delta':own,'outgoing_delta':back,'joint_delta':own+back}
                    break
                p.pose[t][i]=T.copy();p.k[t,i]=row['new_k'];p.z[t]=z;p.C[t][:,i]=col
                for j,cc in bc.items():p.C[1-t][:,j]=cc
    save(root/'source_loss_partition.json',{'rows':rows,'exact_first_c4_event':event,'source_sha':sha(__file__),
        'scope':'Final conditional null-tie classification is not event causality. The named first c4 event is exact trace replay. GT only diagnoses, never updates predictions.'})
    print(__import__('json').dumps({'rows':rows,'event':event},indent=2))


if __name__=='__main__':main()
