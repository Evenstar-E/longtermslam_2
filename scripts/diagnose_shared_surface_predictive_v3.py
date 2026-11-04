"""Post-hoc predictive readout of the sealed VB posterior, no refitting.

E log N used in ELBO is distinct from log E N in predictive density. The latter
uses D+Sigma_mu. This diagnostic does not replace the prespecified primary score.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scripts.run_provenance_pose_hypotheses_v1 import read,save,load,sha
from inspection_decisions.shared_surface_identity_fast_v3 import Model
from inspection_decisions.reversible_surface_v2 import transform
from scripts.prepare_real_temporal_v1 import read_ply


def main():
    root=Path('results/shared_surface_identity_likelihood_v3');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json')
    for f,h in seal['sha256'].items():assert sha(root/f)==h
    predictions=[]
    for scene in cfg['scenes']:
        H,O,_,_,_=load(scene,cfg);disk=np.load(root/scene/'pool.npz');pool=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
        warm=np.load(Path('results/provenance_pose_hypotheses_v1')/scene/'pool_coupled/state.npz');p=Model(H,O,pool,warm,cfg,'shared_vb')
        model=np.load(root/scene/'shared_vb/model.npz');state=np.load(root/scene/'shared_vb/state.npz');ids=np.array([h['uid'] for h in H])
        for t in [0,1]:
            cols=[]
            for i,h in enumerate(H):
                mu=model['mean'][i];var=np.r_[np.zeros((p.kh,6)),model['var'][i]]
                sx=p.sigma[0]**2+var[:,0];sc=p.sigma[3]**2+var[:,3]
                world=transform(mu[:,:3],state['poses'][t,i]);d=cdist(O[t]['xyz'],world,'sqeuclidean')/sx+cdist(O[t]['rgb'],mu[:,3:],'sqeuclidean')/sc
                lp=-.5*d-3*np.log(2*np.pi)-1.5*np.log(sx)-1.5*np.log(sc)+p.component_logweight+p.log_asset
                sem=np.where(O[t]['class']==h['class'],cfg['semantic_correct'],(1-cfg['semantic_correct'])/(p.categories-1));cols.append(logsumexp(lp,axis=1)+np.log(sem))
            L=np.column_stack(cols);a=np.argmax(L,axis=1);value=L[np.arange(len(a)),a];a[value<=p.log_out]=-1
            pred=np.where(a>=0,ids[np.maximum(a,0)],-1)[O[t]['inverse']]
            path=root/scene/f'vb_predictive_period{t+1}.npz';np.savez_compressed(path,identity=pred);predictions.append((scene,t+1,path,H))
    save(root/'predictive_readout_seal.json',{'sha256':{str(p):sha(p) for _,_,p,_ in predictions},'scope':'Post-hoc readout of fixed sealed posterior, after main evaluation. No new training or independent test.'})
    rows=[]
    for scene,period,path,H in predictions:
        if period!=2:continue
        truth=read_ply(Path('data/rescan_third_capture_v1/evaluation/raw')/scene/'gt_segmentation'/f'{scene}_2.ply');pred=np.load(path)['identity']
        eligible={h['uid'] for h in H if h['raw_points']>=128 and h['class'] not in [0,1,2,22]}
        for uid in sorted(eligible):
            use=truth['instance_idx']==uid
            if not use.any():continue
            frac=float(np.mean(pred[use]==uid));rows.append({'scene':scene,'uid':uid,'strict_fraction':frac,'strict_correct':frac>=.8})
    save(root/'predictive_readout_diagnosis.json',{'strict_correct':sum(r['strict_correct'] for r in rows),'total':len(rows),'rows':rows,'scope':'Supplemental fixed-posterior predictive readout, post-hoc diagnostic only. Primary VB results unchanged.'})
    print('PREDICTIVE strict',sum(r['strict_correct'] for r in rows),'/',len(rows))


if __name__=='__main__':main()
