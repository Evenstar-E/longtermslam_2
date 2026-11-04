"""Post-seal, evaluator-only decomposition of a previously identified a9 pose.

The native_4 candidate was identified in the PREVIOUS experiment's GT-assisted
diagnosis. Here it is not selected as a new prediction. All competing poses stay
fixed, so this does not prove a global optimization or model impossibility.
"""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha,load
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem


def main():
    root=Path('results/provenance_pose_hypotheses_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    scene='scene_a';H,O,_,_,_=load(scene,cfg);i=next(j for j,h in enumerate(H) if h['uid']==9)
    disk=np.load(root/scene/'pool.npz');pool=[[disk[f'T_{t+1}_{j}'] for j in range(len(H))] for t in [0,1]]
    meta=next(m for m in read(root/scene/'pool_metadata.json') if m['period']==2 and m['uid']==9);k=meta['order'].index('native_4')
    gtpath=Path('data/rescan_third_capture_v1/evaluation/raw/scene_a/gt_segmentation/scene_a_2.ply')
    truth=read_ply(gtpath)['instance_idx'][O[1]['raw_index']];ids=np.array([h['uid'] for h in H]);output=[]
    for method in ['pool_forward','pool_detached','pool_coupled']:
        state=np.load(root/scene/method/'state.npz');p=Problem(O,H,pool,[state['initial_labels1'],state['initial_labels2']],cfg,method)
        p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.k=state['k'].copy();p.z=[state['labels1'].copy(),state['labels2'].copy()]
        for t in [0,1]:p.C[t]=np.column_stack([p.column(t,j,p.pose[t][j],p.surface(1-t,j) if t==1 or p.bidir else None) for j in range(len(H))])
        T=pool[1][i][k];col=p.column(1,i,T,p.surface(0,i));z,own,_=p.label_proposal(1,i,col)
        back,_,_=p.backward(1,i,T,z) if p.bidir else (0.,{},[])
        C=p.C[1].copy();C[:,i]=col;before=p.assigned(1);after=p.assigned(1,z,C)
        def mapped(a):return np.where(a>=0,ids[np.maximum(a,0)],-1)
        a,b=mapped(p.z[1]),mapped(z);parts=[]
        for uid in np.unique(truth):
            mask=truth==uid;diff=np.dot(p.weights[1][mask],after[mask]-before[mask])
            if abs(diff)>1e-10 or uid==9:
                parts.append({'truth_uid':int(uid),'own_energy_delta':float(diff),'changed_voxels':int(np.sum(mask&(a!=b))),
                    'before_correct_voxels':int(np.sum(mask&(a==uid))),'after_correct_voxels':int(np.sum(mask&(b==uid))),
                    'before_predicted_distribution':{str(int(v)):int(n) for v,n in zip(*np.unique(a[mask],return_counts=True))},
                    'after_predicted_distribution':{str(int(v)):int(n) for v,n in zip(*np.unique(b[mask],return_counts=True))}})
        assert abs(sum(r['own_energy_delta'] for r in parts)-own)<1e-7
        mask=truth==9;mass=O[1]['mass'][mask]
        output.append({'method':method,'selected_k':int(p.k[1,i]),'diagnostic_k':k,'own_delta':own,'outgoing_delta':back,'joint_delta':own+back,
            'a9_selected_pose_support_mass_fraction':float(np.dot(mass,p.C[1][mask,i]<cfg['null_cost'])/mass.sum()),
            'a9_diagnostic_pose_support_mass_fraction':float(np.dot(mass,col[mask]<cfg['null_cost'])/mass.sum()),
            'truth_partition':sorted(parts,key=lambda r:-abs(r['own_energy_delta']))})
    save(root/'rejected_a9_pose_diagnosis.json',{'rows':output,'source_sha':sha(__file__),'truth_sha':sha(gtpath),
        'scope':'GT-assisted conditional failure decomposition at sealed final states; old diagnostic native_4 specified, no reranking or changed predictions. Other asset poses fixed. Voxel representatives/masses, not full raw-point accuracy.'})
    print('REJECTED POSE DIAGNOSED')


if __name__=='__main__':main()
