"""Postseal all192 physical identity+material scoring of restricted refiner."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/sparse_ray_material_incidence_v1_run4');cfg=read(root/'config.json');base=Path(cfg['base_root']);assert read(root/'prediction_seal.json')['complete']
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=[];capacity=[];truth_sources={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_sources[tag]=sha(src/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry'])
        for result in read(root/tag/'predictions.json'):
            i=result['asset'];ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8;material=o+z*r;a=inp['anchors'][i];true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0];ids=bool(result['selected_slots'][0]==truth['slot1'][i] and result['selected_slots'][1]==truth['slot2'][i])
            for method,pred in result['predictions'].items():
                T1=np.array(pred['T1']);T2=np.array(pred['T2']);positions=np.array([transform(a[None],T)[0] for T in [T1,T2]]);error=float(np.linalg.norm(positions[1]-positions[0]-true_d));scan=result['scan'];values=[s[method] for s in scan] if method!='original' else None
                rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'family':result['family'],'method':method,'both_ids':ids,'physical_anchor_error_m':error,'identity_and_material_5cm':bool(ids and error<=.05),'correct_ids_wrong_material':bool(ids and error>.05),'epsilon_m':pred['epsilon_m'],'score_range':float(max(values)-min(values)) if values else None,'score_flat':bool(max(values)-min(values)<=1e-10) if values else None,'raw_late_rays':result['raw_ray_counts'][2],'unavailable_rays':scan[pred['scan_index']]['unavailable_rays'],'availability_invariant':result['availability_invariant_on_scan']})
            T1=np.array(result['initial_T1']);T2=np.array(result['initial_T2']);v=T2[:3,2];old_d=transform(a[None],T2)[0]-transform(a[None],T1)[0];err=old_d-true_d;eps=np.array([s['epsilon_m'] for s in result['scan']]);errors=np.linalg.norm(err[None]+eps[:,None]*v,axis=1);best=np.clip(-err@v,*result['gauge_epsilon_interval_m']);continuous=float(np.linalg.norm(err+best*v))
            capacity.append({'scene':tag,'geometry':run['geometry'],'asset':i,'family':result['family'],'both_ids':ids,'initial_error_m':float(np.linalg.norm(err)),'oracle_pool_min_error_m':float(errors.min()),'oracle_pool_task_success':bool(ids and errors.min()<=.05),'oracle_continuous_line_min_error_m':continuous,'oracle_line_task_success':bool(ids and continuous<=.05),'postseal_evaluation_only':True})
        print('SCORED',tag,flush=True)
    families=sorted({r['family'] for r in rows});methods=['original','nn','plane','sparse_ordered','sparse_unordered'];summary={};paired=[];lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows}
    for mode in ['aligned','offgrid','all']:
        summary[mode]={}
        for family in families:
            summary[mode][family]={}
            for method in methods:
                rr=[r for r in rows if r['family']==family and r['method']==method and (mode=='all' or r['geometry']==mode)];summary[mode][family][method]={'n':len(rr),'both_ids':sum(r['both_ids'] for r in rr),'identity_and_material_5cm':sum(r['identity_and_material_5cm'] for r in rr),'correct_ids_wrong_material':sum(r['correct_ids_wrong_material'] for r in rr),'moved':sum(abs(r['epsilon_m'])>1e-12 for r in rr),'flat_scores':sum(r['score_flat'] is True for r in rr),'median_absolute_move_m':float(np.median([abs(r['epsilon_m']) for r in rr]))}
                for control in ['original','plane','sparse_ordered']:
                    if method==control:continue
                    cc=[lookup[(r['scene'],r['asset'],family,control)] for r in rr];gain=sum(a['identity_and_material_5cm'] and not b['identity_and_material_5cm'] for a,b in zip(rr,cc));loss=sum(b['identity_and_material_5cm'] and not a['identity_and_material_5cm'] for a,b in zip(rr,cc));paired.append({'geometry':mode,'family':family,'method':method,'control':control,'gained':gain,'lost':loss,'net':gain-loss})
    assert summary['all']['catalogue_plane']['original']['identity_and_material_5cm']==115
    assert summary['all']['material_shared']['original']['identity_and_material_5cm']==80
    cap_summary={f:{'n':sum(r['family']==f for r in capacity),'oracle_pool_success':sum(r['family']==f and r['oracle_pool_task_success'] for r in capacity),'oracle_continuous_line_success':sum(r['family']==f and r['oracle_line_task_success'] for r in capacity),'correct_ids_wrong_initial_but_no_line_solution':sum(r['family']==f and r['both_ids'] and r['initial_error_m']>.05 and not r['oracle_line_task_success'] for r in capacity)} for f in families}
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'line_capacity.json',{'summary':cap_summary,'rows':capacity});save(root/'evaluation_sources.json',{'source_sha256':sha(Path(__file__)),'truth_sha256':truth_sources,'GT_read_only_after_full_seal':True,'GT_used_for_inference_or_retuning':False});print(summary,flush=True);print('CAPACITY',cap_summary,flush=True)


if __name__=='__main__':main()
