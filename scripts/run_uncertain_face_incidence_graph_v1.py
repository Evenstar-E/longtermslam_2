"""Full192 x two predicted-ID initializations; inference never reads truth."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from inspection_decisions.uncertain_face_incidence_graph_v1 import extract_pixels,induce_graph,JointProblem
from scripts.check_uncertain_face_incidence_graph_v1 import checks


def main():
    cp=Path('configs/uncertain_face_incidence_graph_v1.json');options=read(cp);old=Path(options['input_root']);cfg={**read(old/'config.json'),**options};out=Path(cfg['output_root']);base=Path(cfg['base_root']);cache=Path(cfg['cached_model_root']);preflight=checks()
    for root in [old,cache]:
        for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight);sources=[cp,Path(__file__),Path('inspection_decisions/uncertain_face_incidence_graph_v1.py'),Path('scripts/check_uncertain_face_incidence_graph_v1.py'),Path('scripts/check_uncertain_face_incidence_measured_jac_v1.py'),Path('research/uncertain_face_incidence_execution_contract_v1_2026-10-25.md')];inputs=[Path('research/uncertain_face_incidence_measured_jac_preflight_v1.json'),old/'prediction_seal.json',old/'config.json',old/'runs.json',cache/'prediction_seal.json',cache/'manifest.json']+list(old.glob('*/predictions.json'))+list(old.glob('*/raw_pairs.pkl'))+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'))
    limits=read(cache/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'motion_limits':limits,'all54_and_ICP_files_retained_not_rerun':True,'identities_fixed':True,'graph_topology_fixed':True,'both_full_SE3_and_shared_faces_updated':True,'not_equal_FLOPs':True});runs=[];start=time.perf_counter()
    for run in read(old/'runs.json'):
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (old/tag/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;data=[]
        for t in range(3):
            frames=[]
            for path in sorted((base/tag).glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                frame['labels'],_=labels(frame,slots,groups,cfg);frames.append(frame)
            data.append([extract_pixels(frames,raw[t][slot],slot,slots[slot],cfg) for slot in range(24)])
        with (dest/'observations.pkl').open('wb') as f:pickle.dump(data,f)
        results=[]
        for source in read(old/tag/'predictions.json'):
            i=source['asset'];s1,s2=source['selected_slots'];xx=np.r_[pose_vector(np.array(source['initial_T1']),slots[s1]),pose_vector(np.array(source['initial_T2']),slots[s2])];track=[data[0][i],data[1][s1],data[2][s2]];graph=induce_graph([(np.zeros(6),track[0]),(xx[:6],track[1])],cfg);predictions={};logs={};parameters={}
            for method in cfg['methods']:
                problem=JointProblem(track,graph,xx,limits,cfg,method);p,log=problem.fit();predictions[method]={'T1':world_pose(p[:6],slots[s1]),'T2':world_pose(p[6:12],slots[s2])};logs[method]=log;parameters[method]=p
            predictions['original']={'T1':source['initial_T1'],'T2':source['initial_T2']};results.append({'asset':i,'family':source['family'],'selected_slots':[s1,s2],'initial_parameters':xx,'graph':graph,'predictions':predictions,'logs':logs,'parameters':parameters})
            if i%6==5:print('PROGRESS',tag,source['family'],i+1,'s',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',results);item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(initializations=len(results),fits=len(results)*len(cfg['methods']),elapsed_s=time.perf_counter()-clock);runs.append(item);save(out/'runs.json',runs);print('DONE',tag,round(item['elapsed_s'],1),flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',flush=True)


if __name__=='__main__':main()
