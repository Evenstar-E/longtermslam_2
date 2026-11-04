"""All 192 mainline observations and old-gauge diagnostics, GT-free."""
from pathlib import Path
import pickle,time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.edge_conditioned_material_support_v1 import extract_edges
from inspection_decisions.set_valued_material_edges_v1 import segment_intervals,minimum_support_cover,optimal_cover_relations,unrestricted_support_feasibility
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from scripts.check_set_valued_material_edges_v1 import checks


def grouped(rows):
    out={}
    for r in rows:out.setdefault(tuple(r['key']),[]).append(r)
    return out


def main():
    cp=Path('configs/edge_conditioned_material_support_v1.json');cfg=read(cp);cache=Path(cfg['cached_model_root']);cfg={**read(cache/'config.json'),**cfg};cfg['output_root']='results/set_valued_material_edges_v1';base=Path(cfg['base_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for root in [base,cache]:
        for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    sources=[cp,Path(__file__),Path('inspection_decisions/set_valued_material_edges_v1.py'),Path('inspection_decisions/edge_conditioned_material_support_v1.py'),Path('scripts/check_set_valued_material_edges_v1.py'),Path('research/set_valued_material_edges_contract_v1_2026-10-25.md'),Path('inspection_decisions/material_supported_continuous_alignment_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py')]
    inputs=[cache/'prediction_seal.json',cache/'manifest.json',base/'runs.json']+list(base.glob('*/frame_*.npz'))+list(base.glob('*/inference.pkl'))+list(cache.glob('*/features.pkl'))+list(cache.glob('*/candidates.pkl'))+list(cache.glob('*/predictions.json'))
    save(out/'config.json',cfg);save(out/'preflight.json',checks());limits=read(cache/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'motion_limits':limits,'task_accuracy_comparison':False})
    runs=[];start=time.perf_counter();all_rows=[];all_groups=[]
    for run in read(base/'runs.json'):
        tag=run['scene'];dest=out/tag;dest.mkdir();src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (cache/tag/'features.pkl').open('rb') as f:features=pickle.load(f)
        with (cache/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;edges=[];logs=[]
        for t in range(3):
            per_slot=[[] for _ in range(24)]
            for v,path in enumerate(sorted(src.glob(f'frame_{t}_*.npz'))):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                frame['edge_view']=v;lab,world=labels(frame,slots,groups,cfg)
                for j in range(24):
                    ff,log=extract_edges(frame,lab,world,j,slots[j],features[t][j],cfg,cfg['seed']+1000*t+100*v+j);per_slot[j].extend(ff);logs.append({'epoch':t,'view':v,'slot':j,**log})
            edges.append(per_slot)
        with (dest/'edges.pkl').open('wb') as f:pickle.dump(edges,f)
        save(dest/'edge_logs.json',logs);selection=read(cache/tag/'predictions.json')['material_shared']['selection'];rows=[];relations=[]
        for i,k,l in selection:
            chosen=[candidates[i][0][k],candidates[i][1][l]];x=np.r_[pose_vector(chosen[0]['T'],slots[chosen[0]['slot']]),pose_vector(chosen[1]['T'],slots[chosen[1]['slot']])];ff=[edges[t+1][b['slot']] for t,b in enumerate(chosen)]
            H=segment_intervals(np.zeros(6),edges[0][i],cfg,0);current=[segment_intervals(x[6*t:6*t+6],f,cfg,t+1) for t,f in enumerate(ff)]
            for key,rr in grouped(H+current[0]+current[1]).items():
                relation=optimal_cover_relations(rr);relations.append({'asset':i,'key':key,'intervals':rr,**relation})
                all_groups.append({'scene':tag,'asset':i,'key':key,'internally_feasible':relation['feasible'],'segments':len(rr),'minimum_supports':relation['minimum_supports'],'relations_resolved':relation['relations_resolved'],'cross_period_possible_pairs':sum(relation['possible_same'][a][b] for a in range(len(rr)) for b in range(a+1,len(rr)) if rr[a]['epoch']!=rr[b]['epoch']) if relation['relations_resolved'] else None,'cross_period_forced_pairs_given_minimum_K':sum(relation['forced_same'][a][b] for a in range(len(rr)) for b in range(a+1,len(rr)) if rr[a]['epoch']!=rr[b]['epoch']) if relation['relations_resolved'] else None})
            R=Rotation.from_rotvec(x[6:9]).as_matrix();v=R[:,2];u=x[9:12];dot=u@v;disc=max(0.,dot*dot+limits[1]**2-u@u);ends=[float(-dot-np.sqrt(disc)),float(-dot+np.sqrt(disc))];scan=[]
            for eps in np.linspace(*ends,17):
                xx=x.copy();xx[9:12]+=eps*v;rr=H+current[0]+segment_intervals(xx[6:],ff[1],cfg,2);gg=grouped(rr);covers=[minimum_support_cover(g) for g in gg.values()];feasible=all(c['feasible'] for c in covers)
                scan.append({'epsilon_m':float(eps),'all_segments_internally_compatible':feasible,'unrestricted_shared_support_feasible':unrestricted_support_feasibility(rr)['feasible'],'minimum_total_supports':sum(c['minimum_supports'] for c in covers) if feasible else None,'incompatible_segments':sum(not r['internally_compatible'] for r in rr)})
            ks=[s['minimum_total_supports'] for s in scan if s['minimum_total_supports'] is not None];compatible=all(s['all_segments_internally_compatible'] for s in scan)
            item={'scene':tag,'asset':i,'selected_current_slots':[b['slot'] for b in chosen],'old_gauge_interval_m':ends,'all_scan_segments_internally_compatible':compatible,'unrestricted_support_feasible_all_gauge_samples':all(s['unrestricted_shared_support_feasible'] for s in scan),'minimum_cardinality_varies_along_gauge':len(set(ks))>1 if compatible else None,'minimum_cardinality_range':[min(ks),max(ks)] if ks else None,'scan':scan};rows.append(item);all_rows.append(item)
        save(dest/'selected_gauge_scan.json',rows);save(dest/'minimum_cover_relations.json',relations);item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(segments=sum(l['segments'] for l in logs),censored_neighbors=sum(l['occlusion_or_unresolved_censored'] for l in logs));runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item,flush=True)
    summary={'asset_conditions':len(all_rows),'observed_segments':sum(r['segments'] for r in runs),'incompatible_selected_direction_groups':sum(not r['internally_feasible'] for r in all_groups),'feasible_selected_direction_groups':sum(r['internally_feasible'] for r in all_groups),'feasible_groups_requiring_multiple_supports':sum(r['internally_feasible'] and r['minimum_supports']>1 for r in all_groups),'all_gauge_samples_locally_compatible_tasks':sum(r['all_scan_segments_internally_compatible'] for r in all_rows),'unrestricted_support_feasible_entire_sampled_gauge_tasks':sum(r['unrestricted_support_feasible_all_gauge_samples'] for r in all_rows),'K_prior_varies_tasks_among_locally_compatible':sum(r['minimum_cardinality_varies_along_gauge'] is True for r in all_rows),'relation_groups_resolved':sum(r['relations_resolved'] for r in all_groups),'relation_groups_unresolved':sum(not r['relations_resolved'] for r in all_groups),'elapsed_s':time.perf_counter()-start,'GT_read':False,'new_end_to_end_task_accuracy_measured':False}
    save(out/'summary.json',summary);save(out/'all_selected_groups.json',all_groups);save(out/'all_gauge_scans.json',all_rows);save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',summary,flush=True)


if __name__=='__main__':main()
