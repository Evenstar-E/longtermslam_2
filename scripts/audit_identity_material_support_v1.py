from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import load,read,sha,save
from inspection_decisions.identity_material_support_v1 import Problem


def main():
    root=Path('results/identity_material_support_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete'];checks=0
    def check(v):
        nonlocal checks
        assert bool(v);checks+=1
    for p,h in seal['sha256'].items():check(sha(root/p)==h)
    for p,h in read(root/'manifest.json')['sources'].items():check(sha(p)==h)
    prior=Path('results/provenance_pose_hypotheses_v1')
    check(sha(prior/'prediction_seal.json')==read(root/'manifest.json')['prior_prediction_seal'])
    for group in ['sources','inputs']:
        for p,h in read(prior/'manifest.json')[group].items():check(sha(p)==h)
    for scene in cfg['scenes']:
        H,O,raw,_,_=load(scene,cfg);disk=np.load(root/scene/'pool.npz');pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
        for method in cfg['methods'][1:]:
            path=root/scene/method;state=np.load(path/'state.npz');m=np.load(path/'material_state.npz');run=read(path/'run.json') if (path/'run.json').exists() else next(r for r in read(root/'runs.json') if r['scene']==scene and r['method']==method)
            p=Problem(O,H,pools,[state['labels1'],state['labels2']],cfg,method=='material_support')
            p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.z=[state['labels1'],state['labels2']];p.M=[m['M1'],m['M2']]
            trace=read(path/'trace.json');check(len(trace)==2*len(H))
            for row in trace:
                check(row['energy_after']<=row['energy_before']+2e-7)
                check(len(row['candidates'])==len(pools[row['period']-1][row['asset_index']]))
            for t in [0,1]:
                p.C[t]=np.column_stack([p.column(t,i,p.pose[t][i],p.surface(1-t,i)) for i in range(len(H))])
                inc=np.load(path/f'source_for{t+1}.npz');seen=[]
                for i in range(len(H)):
                    s=p.surface(1-t,i)
                    for key in ['xyz','rgb','source_voxel']:check(np.array_equal(s[key],inc[f'{key}_{i}']))
                    seen.extend(s['source_voxel'].tolist())
                    check(np.array_equal(state['poses'][t,i],pools[t][i][state['k'][t,i]]))
                check(len(seen)==len(set(seen)));check(O[t]['mass'].sum()==len(raw[t]['xyz']))
                pred=np.load(path/f'prediction{t+1}.npz')['identity'];ids=np.array([h['uid'] for h in H]);z=p.z[t][O[t]['inverse']]
                check(np.array_equal(pred,np.where(z>=0,ids[np.maximum(z,0)],-1)))
                if method=='tie_preserve':check(p.M[t].all())
            check(abs(p.energy()-run['final_energy'])<2e-7)
            print('AUDITED',scene,method,flush=True)
    save(root/'audit.json',{'passed':True,'checks':checks,'scope':'Hashes, finite-pool coverage, monotonic conditional energy, source A/M lineage and all-point readout; not physical truth or posterior calibration.'})
    print('PASS',checks)


if __name__=='__main__':main()
