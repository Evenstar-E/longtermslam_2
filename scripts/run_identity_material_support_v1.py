"""Same sealed starting state, one full-pool refinement sweep; no GT input."""
from pathlib import Path
import shutil,time
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import load,read,save,sha,export
from inspection_decisions.identity_material_support_v1 import Problem


def main():
    prior=Path('results/provenance_pose_hypotheses_v1');seal=read(prior/'prediction_seal.json')
    for p,h in seal['sha256'].items():assert sha(prior/p)==h,p
    out=Path('results/identity_material_support_v1');out.mkdir(exist_ok=False)
    cfg=read(prior/'config.json');cfg['methods']=['warm_start','tie_preserve','material_support'];cfg['sweeps']=1
    cfg['scope']='DEV warm-start refinement, same previous coupled state, one full finite-pool sweep; not native-initialized full pipeline or posterior.'
    save(out/'config.json',cfg)
    files=[Path(__file__),Path('inspection_decisions/identity_material_support_v1.py'),Path('inspection_decisions/provenance_pose_hypotheses_v1.py'),Path('research/identity_material_support_contract_v1_2026-10-24.md')]
    save(out/'manifest.json',{'sources':{str(p):sha(p) for p in files},'prior_prediction_seal':sha(prior/'prediction_seal.json'),'current_GT_read':False})
    runs=[];start=time.perf_counter()
    for scene in cfg['scenes']:
        root=out/scene;root.mkdir();old=prior/scene
        for f in ['assets.json','observations.npz','pool.npz','pool_metadata.json']:shutil.copyfile(old/f,root/f)
        shutil.copytree(old/'pool_coupled',root/'warm_start')
        H,O,raw,_,_=load(scene,cfg);disk=np.load(old/'pool.npz')
        pools=[[disk[f'T_{t+1}_{i}'] for i in range(len(H))] for t in [0,1]]
        state=np.load(old/'pool_coupled/state.npz');labels=[state['labels1'],state['labels2']]
        for method in cfg['methods'][1:]:
            began=time.perf_counter();dest=root/method;dest.mkdir()
            p=Problem(O,H,pools,labels,cfg,method=='material_support')
            p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.k=state['k'].copy()
            p.C=[np.column_stack([p.column(t,i,p.pose[t][i],p.surface(1-t,i)) for i in range(len(H))]) for t in [0,1]]
            p.initial_energy=p.energy();assert abs(p.energy()-read(old/'pool_coupled/run.json')['final_energy'])<2e-7
            def monitor(row,problem):
                small={k:v for k,v in row.items() if k!='candidates'}
                save(out/'monitor.json',{'scene':scene,'method':method,'latest':small,'elapsed_s':time.perf_counter()-start})
                if row.get('asset_index',1)%5==0 or 'sweep_complete' in row:print(scene,method,small,flush=True)
            p.solve(monitor);error=p.verify_cache()
            export(dest,p,raw,[state['initial_labels1'],state['initial_labels2']])
            np.savez_compressed(dest/'material_state.npz',M1=p.M[0],M2=p.M[1],warm_labels1=labels[0],warm_labels2=labels[1])
            row={'scene':scene,'method':method,'initial_energy':p.initial_energy,'final_energy':p.energy(),'cache_error':error,
                 'elapsed_s':time.perf_counter()-began,'queries':p.point_queries,'admitted_voxels':[int(m.sum()) for m in p.M]}
            runs.append(row);save(out/'runs.json',runs);print('DONE',row,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'elapsed_s':time.perf_counter()-start})
    print('SEALED',flush=True)


if __name__=='__main__':main()
