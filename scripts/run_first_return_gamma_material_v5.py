"""No-GT fixed-budget MILP repair and full continuous refits from sealed v4."""
from pathlib import Path
import pickle,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames,refine
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose


def repair(C,previous,cfg):
    old=previous['log'];a,b=np.array(previous['slots']);value=float(C[np.arange(24),a,b].sum());log=dict(old);record={'triggered':False,'old_energy':value,'old_gap':old['gap'],'accepted':False}
    if old['gap'] is not None and old['gap']>cfg['mip_trigger_gap']:
        record['triggered']=True;i,j,k=np.indices(C.shape);col=np.arange(C.size);A=coo_matrix((np.ones(3*C.size),(np.r_[i.ravel(),24+j.ravel(),48+k.ravel()],np.tile(col,3))),shape=(72,C.size)).tocsc();clock=time.perf_counter();r=milp(C.ravel(),integrality=np.ones(C.size),bounds=Bounds(0,1),constraints=LinearConstraint(A,np.ones(72),np.ones(72)),options={'time_limit':cfg['mip_time_limit_s'],'mip_rel_gap':0.});record.update(status=int(r.status),message=str(r.message),elapsed_s=time.perf_counter()-clock)
        bound=getattr(r,'mip_dual_bound',None);record['dual_bound']=float(bound) if bound is not None and np.isfinite(bound) else None
        if r.x is not None and np.max(abs(r.x-np.rint(r.x)))<1e-6:
            ii,jj,kk=np.unravel_index(np.flatnonzero(r.x>.5),C.shape)
            if len(ii)==24 and np.array_equal(ii,np.arange(24)) and sorted(jj)==sorted(kk)==list(range(24)):
                trial=float(C[ii,jj,kk].sum());record['MIP_feasible_energy']=trial
                if trial<value-cfg['mip_improvement_tolerance']:value,a,b=trial,jj,kk;record['accepted']=True
        if record['dual_bound'] is not None:log['LP_lower_bound']=max(log['LP_lower_bound'],record['dual_bound'])
    log.update(energy=value,gap=value-log['LP_lower_bound'],global_optimum_certified=False,standard_MIP_repair=record);return {'slots':np.array([a,b]),'log':log}


def main():
    cp=Path('configs/first_return_gamma_material_v5.json');options=read(cp);geo=Path(options['geometry_root']);assert not (geo/'evaluation_sources.json').exists() and not (geo/'scores.json').exists();seal=read(geo/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8
    for p,h in seal['sha256'].items():assert sha(geo/p)==h
    parent=read(geo/'manifest.json')
    for field in ['sources','inputs']:
        for p,h in parent[field].items():assert sha(p)==h
    cfg={**read(geo/'config.json'),**options};out=Path(cfg['output_root']);out.mkdir(exist_ok=False);sources={**parent['sources'],**{str(p):sha(p) for p in [Path(__file__),cp,Path('research/first_return_gamma_assignment_repair_v5_2026-10-25.md')]}};save(out/'config.json',cfg);save(out/'manifest.json',{'sources':sources,'inputs':{str(geo/'prediction_seal.json'):sha(geo/'prediction_seal.json')},'motion_limits':parent['motion_limits'],'GT_read':False,'only_standard_solver_repair':True,'same_full_candidate_costs':True});runs=[];begin=time.perf_counter()
    for run in read(geo/'runs.json'):
        tag=run['scene'];src=geo/tag;dest=out/tag;dest.mkdir();clock=time.perf_counter();local={**read(src/'scene_config.json'),**options};save(dest/'scene_config.json',local)
        for name in ['workspace_certificate.json','raw_ray_inventory.json']:save(dest/name,read(src/name))
        with (Path(local['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path(local['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];z=np.load(src/'costs.npz');C=z['C'];choices=z['choices'];old=read(src/'assignments.json');assignment={}
        for mi,m in enumerate(cfg['methods']):assignment[m]=repair(C[mi],old[m],cfg);print('MIP',tag,m,assignment[m]['log']['gap'],assignment[m]['log']['standard_MIP_repair'],flush=True)
        save(dest/'assignments.json',assignment);H=np.load(src/'history_stats.npz')['H'];frames,_,_=load_frames(Path(local['base_root']),tag,centers,local);rows=[]
        for i in range(24):
            row={'asset':i,'methods':{}}
            for mi,m in enumerate(cfg['methods']):
                slots=assignment[m]['slots'][:,i];j,k=slots;aa,bb=choices[mi,i,j,k];p1,p2=pools[i][0][aa],pools[i][1][bb];x=np.r_[pose_vector(p1['T'],centers[j]),pose_vector(p2['T'],centers[k])];p,log=refine(H[i],frames[1:],slots,x,centers,parent['motion_limits'],local,m);row['methods'][m]={'candidate_indices':[int(aa),int(bb)],'initial_parameters':x,'final_parameters':p,'log':log,'stages':{'candidate':{'slots':slots,'T1':p1['T'],'T2':p2['T']},'final':{'slots':slots,'T1':world_pose(p[:6],centers[j]),'T2':world_pose(p[6:],centers[k])}}}
            rows.append(row)
        save(dest/'predictions.json',rows);runs.append({**run,'v4_elapsed_s':run['elapsed_s'],'elapsed_s':time.perf_counter()-clock,'continuous_fits':96});save(out/'runs.json',runs);print('V5_DONE',tag,flush=True)
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'scenes':8,'assets':192,'continuous_fits':768,'elapsed_s':time.perf_counter()-begin,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('ALL_V5_SEALED_BEFORE_GT',flush=True)


if __name__=='__main__':main()
