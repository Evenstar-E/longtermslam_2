"""Post-hoc conditional pair intervention, NOT a new evaluated algorithm.

Pair (9,10) and a9/native_4 are selected from GT-assisted failure diagnosis.
Keep a9 at that candidate and enumerate all 27 a10 target candidates. Ranking
uses the frozen joint surrogate only. Other poses/source labels remain fixed.
"""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import load,read,save,sha
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem


def main():
    root=Path('results/provenance_pose_hypotheses_v1');seal=read(root/'prediction_seal.json');assert seal['complete']
    for path,h in seal['sha256'].items():assert sha(root/path)==h
    cfg=read(root/'config.json');H,O,_,_,_=load('scene_a',cfg);ids=np.array([h['uid'] for h in H])
    disk=np.load(root/'scene_a/pool.npz');pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
    state=np.load(root/'scene_a/pool_coupled/state.npz');p=Problem(O,H,pools,[state['initial_labels1'],state['initial_labels2']],cfg,'pool_coupled')
    p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.z=[state['labels1'].copy(),state['labels2'].copy()];p.k=state['k'].copy()
    for t in [0,1]:p.C[t]=np.column_stack([p.column(t,i,p.pose[t][i],p.surface(1-t,i)) for i in range(len(H))])
    i,j=[int(np.flatnonzero(ids==u)[0]) for u in [9,10]]
    meta=next(r for r in read(root/'scene_a/pool_metadata.json') if r['period']==2 and r['uid']==9);ki=meta['order'].index('native_4')
    fixed=pools[1][i][ki];coli=p.column(1,i,fixed,p.surface(0,i));incoming=p.surface(0,j);before=p.assigned(1);records=[];best=None
    for kj,T in enumerate(pools[1][j]):
        colj=p.column(1,j,T,incoming);C=p.C[1].copy();C[:,i]=coli;C[:,j]=colj
        region=np.flatnonzero((p.z[1]==i)|(p.z[1]==j)|(p.C[1][:,i]<p.null)|(p.C[1][:,j]<p.null)|(coli<p.null)|(colj<p.null))
        a=np.argmin(C[region],axis=1);value=C[region,a];a[value>=p.null]=-1;z=p.z[1].copy();z[region]=a
        own=float(np.dot(p.weights[1],p.assigned(1,z,C)-before))
        pp=list(p.pose[1]);pp[i]=fixed;pp[j]=T;changed=z!=p.z[1]
        affected=sorted(({i,j}|set(z[changed])|set(p.z[1][changed]))-{-1});back=0.
        for h in affected:
            cc=p.column(0,h,p.pose[0][h],p.surface(1,h,z,pp));mask=p.z[0]==h
            back+=float(np.dot(p.weights[0][mask],cc[mask]-p.C[0][mask,h]))
        row={'a9_k':ki,'a10_k':kj,'own_delta':own,'outgoing_delta':back,'joint_delta':own+back,'changed_voxels':int(changed.sum())};records.append(row)
        if best is None or row['joint_delta']<best[0]['joint_delta']:best=(row,z.copy())
    # Only after surrogate ranking do we compute descriptive truth changes.
    path=Path('data/rescan_third_capture_v1/evaluation/raw/scene_a/gt_segmentation/scene_a_2.ply')
    truth=read_ply(path)['instance_idx'][O[1]['raw_index']];z=best[1]
    olduid=np.where(p.z[1]>=0,ids[np.maximum(p.z[1],0)],-1);newuid=np.where(z>=0,ids[np.maximum(z,0)],-1)
    target=[]
    for uid in [9,10]:
        mask=truth==uid;mass=O[1]['mass'][mask]
        target.append({'truth_uid':uid,'before_strict_mass_fraction':float(np.dot(mass,olduid[mask]==uid)/mass.sum()),
                       'after_strict_mass_fraction':float(np.dot(mass,newuid[mask]==uid)/mass.sum())})
    save(root/'pair_exchange_diagnosis.json',{'candidates':records,'best_by_joint_surrogate':best[0],'truth_diagnostics':target,
        'source_sha':sha(__file__),'truth_sha':sha(path),
        'scope':'Post-hoc GT-informed pair/candidate selection, conditional diagnostic only; NOT unprivileged new algorithm or additional success count. All a10 pool entries tested; other poses and source labels fixed. Voxel representative mass, not full raw-point metric.'})
    print(__import__('json').dumps({'best':best[0],'truth':target},indent=2))


if __name__=='__main__':main()
