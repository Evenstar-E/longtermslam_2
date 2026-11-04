"""Replay one sealed VB decision; remove only variance for a conditional check."""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,load,sha
from inspection_decisions.shared_surface_identity_fast_v3 import Model


def main():
    root=Path('results/shared_surface_identity_likelihood_v3');seal=read(root/'prediction_seal.json');assert seal['complete']
    for path,h in seal['sha256'].items():assert sha(root/path)==h
    cfg=read(root/'config.json');H,O,_,_,_=load('scene_a',cfg);disk=np.load(root/'scene_a/pool.npz');pool=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
    warm=np.load('results/provenance_pose_hypotheses_v1/scene_a/pool_coupled/state.npz');p=Model(H,O,pool,warm,cfg,'shared_vb');p.update_surface()
    trace=read(root/'scene_a/shared_vb/trace.json');target=next(i for i,h in enumerate(H) if h['uid']==24)
    for row in trace:
        if row['kind']!='pose':continue
        t,i=row['period']-1,row['asset_index'];assert abs(p.objective()-row['before'])<1e-5
        if t==1 and i==target:break
        k=row['selected_k'];p.pose[t,i]=pool[t][i][k];p.k[t,i]=k;p.L[t][:,i]=p.asset_column(t,i)
        assert abs(p.objective()-row['after'])<1e-5
    results=[]
    for mode in ['shared_vb','shared_mle']:
        p.method=mode;L=p.columns(1);other=L.copy();other[:,target]=-np.inf;Z=p.partition(other)
        values=[]
        for k,T in enumerate(pool[1][target]):values.append(float(np.dot(O[1]['mass'],np.logaddexp(Z,p.asset_column(1,target,T)))))
        results.append({'conditional_scoring':mode,'best_k':int(np.argmax(values)),'all_values':values,'k5_minus_k0':values[5]-values[0]})
    save(root/'vb_a24_decision_diagnosis.json',{'rows':results,'sealed_decision_k':row['selected_k'],'replay_exact':True,'source_sha':sha(__file__),
        'scope':'At the exact pre-a24 VB state, freeze means and all other poses; remove posterior variance only from expected-log kernels. KL constant in this pose comparison. Conditional evaluator intervention, no new predictions; not overall VB-vs-MAP causal attribution.'})
    print(__import__('json').dumps([{k:v for k,v in x.items() if k!='all_values'} for x in results],indent=2))


if __name__=='__main__':main()
