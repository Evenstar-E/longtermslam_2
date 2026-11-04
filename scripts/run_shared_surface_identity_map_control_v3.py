"""Joint-likelihood finite-pool experiment. No current truth access."""
from pathlib import Path
import time,shutil
import numpy as np
from scipy.spatial.distance import cdist
from scripts.run_provenance_pose_hypotheses_v1 import load,read,save,sha
from inspection_decisions.shared_surface_identity_fast_v3 import MAPModel as Model
from inspection_decisions.reversible_surface_v2 import transform,inverse


def export(dest,p,raw,state):
    ids=np.array([h['uid'] for h in p.H]);labels=[p.readout(t)[0] for t in [0,1]]
    np.savez_compressed(dest/'state.npz',poses=p.pose,k=p.k,labels1=labels[0],labels2=labels[1],initial_labels1=state['initial_labels1'],initial_labels2=state['initial_labels2'],anchors_world=np.array([[transform(h['anchor'][None],T)[0] for h,T in zip(p.H,pp)] for pp in p.pose]))
    np.savez_compressed(dest/'model.npz',mean=p.mean,var=p.var,prior=p.prior,low=p.low,high=p.high,L1=p.L[0],L2=p.L[1])
    save(dest/'trace.json',p.trace)
    stats=np.zeros((2,p.m,p.ku))
    for t in [0,1]:
        for i in range(p.m):stats[t,i]=np.sum(np.exp(p.components(t,i)[:,p.kh:]-p.partition(p.L[t])[:,None])*p.O[t]['mass'][:,None],axis=0)
        chosen=labels[t][p.O[t]['inverse']];identity=np.where(chosen>=0,ids[np.maximum(chosen,0)],-1)
        out={k:[] for k in ['readout_identity','confidence','readout_confidence','supported','history_supported','readout_supported']}
        confidence=p.readout(t)[1][p.O[t]['inverse']]
        for start in range(0,len(chosen),12000):
            sl=slice(start,start+12000);o={'xyz':raw[t]['xyz'][sl],'rgb':raw[t]['rgb'][sl].astype(float)/255,'class':raw[t]['predicted_class'][sl]}
            a,conf,rs=p.readout(t,o);sup=np.zeros(len(a),bool);hs=sup.copy();z=chosen[sl]
            for i in np.unique(z[z>=0]):
                idx=np.flatnonzero(z==i);world=transform(p.mean[i,:,:3],p.pose[t,i]);sup[idx]=np.min(cdist(o['xyz'][idx],world),axis=1)<p.cfg['distance_scale_m']
                hs[idx]=p.H[i]['tree'].query(inverse(o['xyz'][idx],p.pose[t,i]),workers=1)[0]<p.cfg['distance_scale_m']
            values={'readout_identity':np.where(a>=0,ids[np.maximum(a,0)],-1),'confidence':confidence[sl],'readout_confidence':conf,'supported':sup,'history_supported':hs,'readout_supported':rs}
            for k,v in values.items():out[k].append(v)
        np.savez_compressed(dest/f'prediction{t+1}.npz',identity=identity,**{k:np.concatenate(v) for k,v in out.items()})
    np.savez_compressed(dest/'component_responsibility_mass.npz',mass=stats)


def main():
    prior=Path('results/provenance_pose_hypotheses_v1');seal=read(prior/'prediction_seal.json')
    for f,h in seal['sha256'].items():assert sha(prior/f)==h
    cfgpath=Path('configs/shared_surface_identity_likelihood_v1.json');cfg=read(cfgpath);cfg['methods']=['shared_map']
    root=Path('results/shared_surface_identity_map_control_v3');root.mkdir(exist_ok=False);save(root/'config.json',cfg)
    paths=[Path('research/shared_surface_identity_acceleration_v3_2026-10-24.md'),Path('inspection_decisions/shared_surface_identity_fast_v3.py'),Path('inspection_decisions/shared_surface_identity_fast_v2.py'),Path('research/shared_surface_identity_acceleration_v2_2026-10-24.md'),Path('inspection_decisions/shared_surface_identity_map_control_v1.py'),Path('research/shared_surface_map_control_amendment_v1_2026-10-24.md'),cfgpath,Path(__file__),Path('inspection_decisions/shared_surface_identity_likelihood_v1.py'),Path('research/shared_surface_identity_likelihood_contract_v1_2026-10-24.md'),Path('scripts/run_provenance_pose_hypotheses_v1.py'),Path('scripts/run_rescan_reversible_surface_v2.py'),Path('inspection_decisions/reversible_surface_v2.py')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in paths},'prior_prediction_seal':sha(prior/'prediction_seal.json'),'current_GT_read':False})
    begin=time.perf_counter();runs=[]
    for scene in cfg['scenes']:
        old=prior/scene;dest=root/scene;dest.mkdir()
        for f in ['assets.json','observations.npz','pool.npz','pool_metadata.json']:shutil.copyfile(old/f,dest/f)
        H,O,raw,_,_=load(scene,cfg);disk=np.load(old/'pool.npz');pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]];state=np.load(old/'pool_coupled/state.npz')
        for method in cfg['methods']:
            folder=dest/method;folder.mkdir();start=time.perf_counter();p=Model(H,O,pools,state,cfg,method);initial=p.objective();assert p.tail_bound()<1e-7
            def monitor(row):
                small={k:v for k,v in row.items() if k!='candidate_values'};small.update(scene=scene,method=method,elapsed_s=time.perf_counter()-begin)
                save(root/'monitor.json',small)
                if row['asset_index']%5==0:print(small,flush=True)
            for _ in range(cfg['rounds']):p.update_surface();p.pose_sweep(monitor);p.update_surface()
            error=max(float(np.max(np.abs(p.L[t]-p.columns(t)))) for t in [0,1]);assert error<1e-7
            export(folder,p,raw,state)
            row={'scene':scene,'method':method,'initial_objective':initial,'final_objective':p.objective(),'kl':p.kl(),'tail_loglikelihood_bound':p.tail_bound(),'cache_error':error,'elapsed_s':time.perf_counter()-start,'kernel_evaluations':p.queries}
            save(folder/'run.json',row);runs.append(row);save(root/'runs.json',runs);print('DONE',row,flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-begin,'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED MAP TWO',flush=True)


if __name__=='__main__':main()
