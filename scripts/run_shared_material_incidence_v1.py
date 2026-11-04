"""Run frozen-graph shared-pixel association on ALL old candidates, GT-free."""
from pathlib import Path
import pickle,time
import numpy as np
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.shared_material_incidence_v1 import induce_graph,score_graph
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.check_shared_material_incidence_v1 import checks


def main():
    cp=Path('configs/shared_material_incidence_v1.json');options=read(cp);src=Path(options['input_root']);out=Path(options['output_root']);cfg={**read(src/'config.json'),**options}
    assert read(src/'prediction_seal.json')['complete']
    for name,h in read(src/'prediction_seal.json')['sha256'].items():assert sha(src/name)==h
    preflight=checks();out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight)
    sources=[cp,Path(__file__),Path('scripts/check_shared_material_incidence_v1.py'),Path('inspection_decisions/shared_material_incidence_v1.py'),Path('research/shared_material_incidence_implementation_contract_v1_2026-10-25.md')]
    inputs=[src/'prediction_seal.json',src/'config.json',src/'runs.json']+list(src.glob('*/raw_pairs.pkl'))+list(src.glob('*/predictions.json'))+list(Path(cfg['base_root']).glob('*/inference.pkl'))
    save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'identities_rotations_C1_frozen':True,'graph_offsets_frozen_after_reference_induction':True,'new_instance_birth_not_implemented':True,'same_candidates_not_equal_FLOPs':True})
    runs=[];start=time.perf_counter()
    for run in read(src/'runs.json'):
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (src/tag/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];results=[];graphs=[]
        for result in read(src/tag/'predictions.json'):
            i=result['asset'];s1,s2=result['selected_slots'];x1=pose_vector(np.array(result['initial_T1']),centers[s1]);x2=pose_vector(np.array(result['initial_T2']),centers[s2]);graph=induce_graph([(np.zeros(6),raw[0][i]),(x1,raw[1][s1])],cfg);graphs.append({'asset':i,'family':result['family'],'graph':graph});direction=np.array(result['initial_T2'])[:3,2]
            for row in result['scan']:
                x=x2.copy();x[3:]+=row['epsilon_m']*direction;cost=score_graph(x,raw[2][s2],graph,cfg)
                for key in ['independent','shared','shared_no_domain']:row['graph_'+key]=cost[key]
                row['graph_diagnostics']={k:v for k,v in cost.items() if k not in ['independent','shared','shared_no_domain']}
            for method in ['graph_independent','graph_shared','graph_shared_no_domain']:
                minimum=min(s[method] for s in result['scan']);eligible=[j for j,s in enumerate(result['scan']) if s[method]<=minimum+cfg['selection_tolerance']];j=min(eligible,key=lambda j:abs(result['scan'][j]['epsilon_m']));eps=result['scan'][j]['epsilon_m'];T2=np.array(result['initial_T2']);T2[:3,3]+=eps*direction
                result['predictions'][method]={'T1':result['initial_T1'],'T2':T2,'epsilon_m':eps,'scan_index':j}
            result['graph_counts']={'faces':len(graph['faces']),'edges':len(graph['edges']),'reference_pixels':graph['unique_reference_pixels']};results.append(result)
            if i%6==5:print('PROGRESS',tag,result['family'],i+1,'seconds',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',results);save(dest/'graphs.json',graphs);item={**run,'shared_graph_elapsed_s':time.perf_counter()-clock};runs.append(item);save(out/'runs.json',runs);print('DONE',tag,round(item['shared_graph_elapsed_s'],1),flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',flush=True)


if __name__=='__main__':main()
