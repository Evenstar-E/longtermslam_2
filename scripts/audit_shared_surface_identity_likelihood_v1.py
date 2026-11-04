from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts.run_provenance_pose_hypotheses_v1 import load,read,save,sha
from inspection_decisions.shared_surface_identity_likelihood_v1 import Model
from inspection_decisions.reversible_surface_v2 import transform


def main():
    root=Path('results/shared_surface_identity_likelihood_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete'];checks=0;rows=[]
    def check(v):
        nonlocal checks
        assert bool(v);checks+=1
    for p,h in seal['sha256'].items():check(sha(root/p)==h)
    for p,h in read(root/'manifest.json')['sources'].items():check(sha(p)==h)
    prior=Path('results/provenance_pose_hypotheses_v1');check(sha(prior/'prediction_seal.json')==read(root/'manifest.json')['prior_prediction_seal'])
    for group in ['sources','inputs']:
        for p,h in read(prior/'manifest.json')[group].items():check(sha(p)==h)
    for scene in cfg['scenes']:
        H,O,raw,_,_=load(scene,cfg);disk=np.load(root/scene/'pool.npz');pool=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]];warm=np.load(prior/scene/'pool_coupled/state.npz')
        for method in cfg['methods']:
            dest=root/scene/method;state=np.load(dest/'state.npz');model=np.load(dest/'model.npz');run=read(dest/'run.json');trace=read(dest/'trace.json')
            p=Model(H,O,pool,warm,cfg,method);initial_mean=p.mean.copy()
            p.mean=model['mean'];p.var=model['var'];p.prior=model['prior'];p.pose=state['poses'];p.k=state['k'];p.L=[model['L1'],model['L2']]
            check(np.array_equal(p.mean[:,:p.kh],initial_mean[:,:p.kh]));check(np.allclose(np.exp(p.component_logweight).sum(),1))
            check(abs((1-cfg['outlier_mass'])+cfg['outlier_mass']-1)<1e-12)
            check(abs(p.objective()-run['final_objective'])<1e-5)
            poses=[r for r in trace if r['kind']=='pose'];check(len(poses)==2*len(H)*cfg['rounds'])
            prev=run['initial_objective']
            for row in trace:
                check(abs(row['before']-prev)<1e-5);check(row['after']>=prev-1e-5);prev=row['after']
                if row['kind']=='pose':check(len(row['candidate_values'])==len(pool[row['period']-1][row['asset_index']]))
            check(abs(prev-run['final_objective'])<1e-5)
            if method=='frozen_shared':check(np.array_equal(p.mean,initial_mean))
            if method=='shared_vb':check(np.all(p.var>0) and p.kl()>=-1e-7)
            stats=np.load(dest/'component_responsibility_mass.npz')['mass']
            for t in [0,1]:
                check(np.allclose(p.columns(t),p.L[t],atol=1e-7,rtol=0))
                lp=np.c_[p.L[t],np.full(len(O[t]['xyz']),p.log_out)];check(np.allclose(np.exp(lp-logsumexp(lp,axis=1)[:,None]).sum(1),1))
                z=p.readout(t)[0];check(np.array_equal(z,state[f'labels{t+1}']))
                ids=np.array([h['uid'] for h in H]);a=z[O[t]['inverse']];pred=np.load(dest/f'prediction{t+1}.npz')
                check(np.array_equal(pred['identity'],np.where(a>=0,ids[np.maximum(a,0)],-1)));check(len(pred['readout_identity'])==len(raw[t]['xyz']))
                check(O[t]['mass'].sum()==len(raw[t]['xyz']));check(stats[t].sum()<=O[t]['mass'].sum()+1e-5)
                for i,h in enumerate(H):
                    check(np.array_equal(p.pose[t,i],pool[t][i][state['k'][t,i]]))
                    check(np.allclose(state['anchors_world'][t,i],transform(h['anchor'][None],p.pose[t,i])[0]))
                    expected=np.sum(np.exp(p.components(t,i)[:,p.kh:]-p.partition(p.L[t])[:,None])*O[t]['mass'][:,None],axis=0)
                    check(np.allclose(expected,stats[t,i],atol=1e-5))
            rows.append({'scene':scene,'method':method,'objective':p.objective(),'kl':p.kl()});print('AUDITED',scene,method,flush=True)
    save(root/'audit.json',{'passed':True,'checks':checks,'rows':rows,'scope':'Normalized joint-model implementation, source inputs, conditional objective consistency; not original theory, calibration, visibility validity or physical displacement truth.'})
    print('PASS',checks)


if __name__=='__main__':main()
