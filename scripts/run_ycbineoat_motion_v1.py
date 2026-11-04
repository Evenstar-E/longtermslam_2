"""Inference never reads pose annotations; sealed before independent-motion scoring."""
import time
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem
from inspection_decisions.real_temporal_v1 import prepare,register
from inspection_decisions.livingscenes_cpu_runtime_v1 import load_solver

def main():
    cfgpath=Path('configs/ycbineoat_motion_execution_v1.json');cfg=read(cfgpath);root=Path(cfg['public_root']);public=read(root/'public.json')
    out=Path('results/ycbineoat_motion_v1');out.mkdir(exist_ok=False)
    source=[Path(__file__),cfgpath,Path('configs/rigid_identity_bounds_v1.json'),Path('inspection_decisions/rigid_identity_cached_v1.py'),Path('inspection_decisions/rigid_identity_bounds_v1.py'),Path('inspection_decisions/real_temporal_v1.py'),Path('inspection_decisions/livingscenes_cpu_runtime_v1.py')]
    save(out/'start_manifest.json',{'sources_sha256':{str(p):sha(p) for p in source},'public_sha256':{str(p):sha(p) for p in (root/'public').rglob('*') if p.is_file()},'metadata_sha256':sha(root/'public.json'),'GT_pose_read':False,'configuration_sha256':sha(cfgpath)})
    solver=load_solver(Path.cwd(),dict(read('configs/livingscenes_real_v1.json'),torch_threads=1));import torch
    clouds={}
    for row in public['clouds']:
        with np.load(root/row['path']) as z:clouds[row['stem']]={'token':row['stem'],'xyz':z['xyz'],'rgb':z['rgb']}
    records=[];began=time.perf_counter()
    for k,pair in enumerate(public['pairs']):
        H=clouds[pair['history_stem']];C=clouds[pair['current_stem']];dest=out/f'pair_{k:02d}';dest.mkdir()
        problem=CachedJointProblem([H],[C],np.zeros((1,1)),read('configs/rigid_identity_bounds_v1.json'))
        a,fa=prepare(H['xyz'],cfg['geometry']);b,fb=prepare(C['xyz'],cfg['geometry']);methods={}
        T=np.eye(4);T[:3,3]=C['xyz'].mean(0)-H['xyz'].mean(0);methods['centroid_translation']=T
        for method in ['centroid_icp','fgr_gicp']:
            T,stats=register(a,b,fa,fb,method,cfg['geometry'],dest/method);methods[method]=T
        with torch.no_grad():R,t=solver._solve_pairwise_registration(torch.tensor(H['xyz'],dtype=torch.float32)[None],torch.tensor(C['xyz'],dtype=torch.float32)[None],optim=False)
        T=np.eye(4);T[:3,:3]=R.numpy()[0];T[:3,3]=t.numpy()[0,:,0];methods['more_pairwise']=T
        starts=[]
        for name,T in methods.items():
            if T is None:continue
            starts.append({'method':name,'T_current_from_history':T.tolist(),'surrogate_energy':problem.energy(np.array([0]),[T]),'inside_declared_domain':problem.in_domain(T,0)})
        feasible=[r for r in starts if r['inside_declared_domain']];best=min(feasible,key=lambda r:r['surrogate_energy'])
        initial={'assignment':[0],'poses':[best['T_current_from_history']],'energy':best['surrogate_energy']}
        bounded=problem.solve_bounds(initial,lambda r:save(dest/'monitor.json',r));save(dest/'bounded.json',bounded)
        save(dest/'predictions.json',{'pair':pair,'methods':starts,'candidate':bounded['best'],'historical_observed_anchor_m':problem.anchor[0].tolist(),'scope':cfg['scope']})
        records.append({'pair':k,'elapsed_s':time.perf_counter()-began,'candidate_energy':bounded['best']['energy'],'bound_lb':bounded['last_lower_bound']});save(out/'monitor.json',records[-1]);print('MOTION_PAIR_DONE',k,records[-1],flush=True)
    save(out/'prediction_seal.json',{'complete':True,'pairs':len(public['pairs']),'elapsed_s':time.perf_counter()-began,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*.json') if p.name!='prediction_seal.json'},'GT_pose_read':False})

if __name__=='__main__':main()
