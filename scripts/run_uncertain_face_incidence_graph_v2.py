"""Same v1 model, standard Schur repair and common continuous endpoint pools."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from inspection_decisions.material_supported_continuous_alignment_v1 import world_pose
from inspection_decisions.uncertain_face_incidence_graph_v1 import JointProblem
from inspection_decisions.uncertain_face_incidence_schur_v2 import fit,map_parameters
from scripts.check_uncertain_face_incidence_schur_v2 import checks


def choices_for(problem,pool):
    choices=[]
    for name,source,x in pool:
        q=map_parameters(source,problem,x);r,_,_=problem.evaluate(q,False);choices.append((float(r@r),name,q))
        if problem.free_depth and np.any(q[problem.depth_start:]!=problem.z):
            qq=q.copy();qq[problem.depth_start:]=problem.z;r,_,_=problem.evaluate(qq,False);choices.append((float(r@r),name+'@measured_depth',qq))
    return choices


def main():
    src=Path(read('configs/uncertain_face_incidence_graph_v1.json')['output_root']);cfg=read(src/'config.json');assert read(src/'prediction_seal.json')['complete'];out=Path('results/uncertain_face_incidence_graph_v2');cfg={**cfg,'output_root':str(out),'parent_root':str(src),'schur_polish_iterations':8};out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',checks());manifest=read(src/'manifest.json');inputs=[src/'prediction_seal.json',src/'config.json',src/'manifest.json']+list(src.glob('*/observations.pkl'))+list(src.glob('*/predictions.json'));sources=[Path(__file__),Path('inspection_decisions/uncertain_face_incidence_schur_v2.py'),Path('scripts/check_uncertain_face_incidence_schur_v2.py'),Path('research/uncertain_face_incidence_solver_amendment_v2_2026-10-25.md')];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'motion_limits':manifest['motion_limits'],'unchanged_v1_energy':True,'common_pose_endpoints':9,'equal_FLOPs':False});runs=[];start=time.perf_counter()
    for run in read(src/'runs.json'):
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (src/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];results=[]
        for row in read(src/tag/'predictions.json'):
            i=row['asset'];s1,s2=row['selected_slots'];track=[data[0][i],data[1][s1],data[2][s2]];problems={m:JointProblem(track,row['graph'],row['initial_parameters'],manifest['motion_limits'],cfg,m) for m in cfg['methods']};pool=[('original',problems['normal_joint'],problems['normal_joint'].start())]+[(m,problems[m],np.array(row['parameters'][m])) for m in cfg['methods']];fitted={};logs={};start_sources={}
            for m,p in problems.items():
                value,name,q=min(choices_for(p,pool),key=lambda x:x[0]);pp,log=fit(p,q,cfg['schur_polish_iterations']);fitted[m]=pp;logs[m]=log;start_sources[m]=name
            pool.extend([('schur_'+m,problems[m],fitted[m]) for m in cfg['methods']]);predictions={'original':row['predictions']['original']};parameters={};selection={};own_predictions={}
            for m,p in problems.items():
                choices=choices_for(p,pool);value,name,q=min(choices,key=lambda x:x[0]);parameters[m]=q;predictions[m]={'T1':world_pose(q[:6],centers[s1]),'T2':world_pose(q[6:12],centers[s2])};selection[m]={'source':name,'energy':value,'pool_size':len(pool),'latent_depth_variants':len(choices)};pp=fitted[m];own_predictions[m]={'T1':world_pose(pp[:6],centers[s1]),'T2':world_pose(pp[6:12],centers[s2])}
            results.append({**row,'predictions':predictions,'parameters':parameters,'logs':logs,'fit_parameters':fitted,'own_fit_predictions':own_predictions,'common_pool_selection':selection,'start_sources':start_sources})
            if i%6==5:print('PROGRESS',tag,row['family'],i+1,'s',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',results);item={**run,'schur_elapsed_s':time.perf_counter()-clock};runs.append(item);save(out/'runs.json',runs);print('DONE',tag,round(item['schur_elapsed_s'],1),flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',flush=True)


if __name__=='__main__':main()
