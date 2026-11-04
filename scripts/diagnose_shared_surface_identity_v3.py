"""Post-seal ownership and confidence diagnosis, not physical material truth."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts.run_provenance_pose_hypotheses_v1 import read,save,load
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.shared_surface_identity_fast_v3 import Model,MAPModel


def main():
    roots=[Path('results/shared_surface_identity_likelihood_v3'),Path('results/shared_surface_identity_map_control_v3')]
    assert all(read(r/'prediction_seal.json')['complete'] for r in roots)
    rows=[];pairs=[]
    for root in roots:
        cfg=read(root/'config.json')
        for scene in cfg['scenes']:
            H,O,_,_,_=load(scene,cfg);ids=np.array([h['uid'] for h in H]);eligible={h['uid'] for h in H if h['raw_points']>=128 and h['class'] not in [0,1,2,22]}
            disk=np.load(root/scene/'pool.npz');pool=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]];warm=np.load(Path('results/provenance_pose_hypotheses_v1')/scene/'pool_coupled/state.npz')
            truths=[];equivs=[]
            for t in [0,1]:
                base=Path('data/real_cross_session_v1/rescan/raw') if t==0 else Path('data/rescan_third_capture_v1/evaluation/raw')
                truths.append(read_ply(base/scene/'gt_segmentation'/f'{scene}_{t+1}.ply')['instance_idx'][O[t]['raw_index']])
                eq={};file=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_{t+1}.txt'
                if file.exists():
                    for line in file.read_text(encoding='utf-8').splitlines():
                        if line.strip():
                            a,b=line.split('|');eq[int(a)]={int(v) for v in b.split()}
                equivs.append(eq)
            for method in cfg['methods']:
                p=(MAPModel if method=='shared_map' else Model)(H,O,pool,warm,cfg,method);dest=root/scene/method
                state=np.load(dest/'state.npz');model=np.load(dest/'model.npz');p.mean=model['mean'];p.var=model['var'];p.prior=model['prior'];p.pose=state['poses'];p.L=[model['L1'],model['L2']]
                for t in [0,1]:
                    truth=truths[t];eq=equivs[t];use=np.isin(truth,list(eligible));weight=O[t]['mass']*use;Z=p.partition(p.L[t]);good=bad=ngood=nbad=0.
                    for i,uid in enumerate(ids):
                        allowed=np.array([int(uid) in (eq.get(int(v),{int(v)})&eligible) for v in truth])
                        q=np.exp(p.L[t][:,i]-Z);nq=np.exp(logsumexp(p.components(t,i)[:,p.kh:],axis=1)-Z)
                        good+=float(np.dot(weight*allowed,q));bad+=float(np.dot(weight*~allowed,q));ngood+=float(np.dot(weight*allowed,nq));nbad+=float(np.dot(weight*~allowed,nq))
                    unknown=float(np.dot(weight,np.exp(p.log_out-Z)));assert abs(good+bad+unknown-weight.sum())<1e-5
                    a,confidence,_=p.readout(t);pred=np.where(a>=0,ids[np.maximum(a,0)],-1)
                    correct=np.array([int(u) in (eq.get(int(v),{int(v)})&eligible) for u,v in zip(pred,truth)])
                    high=use&(confidence>=.9);highmass=float(O[t]['mass'][high].sum());wrongmass=float(O[t]['mass'][high&~correct].sum())
                    rows.append({'scene':scene,'method':method,'period':t+1,'persistent_mass':float(weight.sum()),'expected_correct_identity_mass':good,'expected_wrong_identity_mass':bad,'expected_outlier_mass':unknown,'expected_newcomponent_correct_mass':ngood,'expected_newcomponent_wrong_mass':nbad,'high_confidence_mass':highmass,'high_confidence_wrong_mass':wrongmass,'conditional_center_position_std_mm_median':float(np.median(np.sqrt(p.var[:,:,0]))*1000) if method=='shared_vb' else None})
                if scene=='scene_a':
                    i,j=[int(np.flatnonzero(ids==u)[0]) for u in [9,10]]
                    meta=next(v for v in read(root/scene/'pool_metadata.json') if v['period']==2 and v['uid']==9)
                    ki=meta['order'].index('native_4');col=p.asset_column(1,i,pool[1][i][ki]);before=float(np.dot(O[1]['mass'],p.partition(p.L[1])));best=None
                    for kj,T in enumerate(pool[1][j]):
                        trial=p.L[1].copy();trial[:,i]=col;trial[:,j]=p.asset_column(1,j,T);value=float(np.dot(O[1]['mass'],p.partition(trial)))
                        if best is None or value>best[0]:best=(value,kj,trial)
                    z=np.argmax(best[2],axis=1);z[np.max(best[2],axis=1)<=p.log_out]=-1;pred=np.where(z>=0,ids[np.maximum(z,0)],-1)
                    coverage={}
                    for uid in [9,10]:
                        mask=truths[1]==uid;coverage[str(uid)]=float(np.dot(O[1]['mass'][mask],pred[mask]==uid)/O[1]['mass'][mask].sum())
                    pairs.append({'method':method,'a9_k':ki,'a10_k':best[1],'joint_score_delta_fixed_surface':best[0]-before,'strict_mass_coverage':coverage})
                print('DIAGNOSED',scene,method,flush=True)
    save(roots[0]/'ownership_diagnosis.json',{'rows':rows,'scope':'Representative-truth weighted responsibility mass, not exact raw contamination, independent trials, material correspondences or calibrated pose/anchor uncertainty.'})
    save(roots[0]/'pair_ranking_diagnosis.json',{'rows':pairs,'scope':'Post-hoc a9/10 pair fixed from previous GT diagnosis, not unprivileged algorithm. Surface posterior and other poses held fixed; native_4 for9, all27partner poses tested. No predictions changed or extra success counted.'})


if __name__=='__main__':main()
