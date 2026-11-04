"""Evaluate all identity stages only AFTER complete eight-scene prediction seal."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.run_typed_material_incidence_graph_v2 import mean_colors


def main():
    root=Path('results/typed_material_incidence_graph_v2');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    assert read(root/'audit.json')['passed'];rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry'])
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        colors=mean_colors(Path(cfg['base_root']),tag,data)
        with (root/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        for result in read(root/tag/'predictions.json'):
            i=result['asset'];ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i];true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0]
            predictions=[('original','original',result['original'])]+[(m,s,p) for m,v in result['methods'].items() for s,p in v['stages'].items()]
            for method,stage,pred in predictions:
                ids=bool(pred['slots'][0]==truth['slot1'][i] and pred['slots'][1]==truth['slot2'][i]);positions=np.array([transform(anchor[None],np.array(pred[k]))[0] for k in ['T1','T2']]);error=float(np.linalg.norm(positions[1]-positions[0]-true_d));color_compatible=all(float(np.max(abs(colors[0,i]-colors[t+1,s])))<=2. for t,s in enumerate(pred['slots']));retained=all(any(b['slot']==s for b in pools[i][t][:108]) for t,s in enumerate(pred['slots']));rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'family':result['family'],'method':method,'stage':stage,'slots':pred['slots'],'both_ids':ids,'error_m':error,'success':bool(ids and error<=.05),'both_slots_in_original108':retained,'both_RGB_compatible_with_SIM_sensor_bound':color_compatible})
        print('SCORED',tag,flush=True)
    lookup={(r['scene'],r['asset'],r['family'],r['method'],r['stage']):r for r in rows};summary={};paired=[];breakdown=[]
    for family in ['catalogue_plane','material_shared']:
        summary[family]={}
        for method,stages in [('original',['original'])]+[(m,['conditional','catalogue','pool','final']) for m in cfg['methods']]:
            summary[family][method]={}
            for stage in stages:
                rr=[r for r in rows if r['family']==family and r['method']==method and r['stage']==stage];errors=[r['error_m'] for r in rr if r['both_ids']];summary[family][method][stage]={'n':len(rr),'success':sum(r['success'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'correct_ID_median_error_m':float(np.median(errors)) if errors else None,'outside_original108_slot_tasks':sum(not r['both_slots_in_original108'] for r in rr),'SIM_RGB_incompatible_tasks':sum(not r['both_RGB_compatible_with_SIM_sensor_bound'] for r in rr)}
                for g in ['aligned','offgrid']:
                    for h in [1,3]:
                        sub=[r for r in rr if r['geometry']==g and r['history_views']==h];breakdown.append({'family':family,'method':method,'stage':stage,'geometry':g,'history_views':h,'n':len(sub),'success':sum(r['success'] for r in sub),'both_ids':sum(r['both_ids'] for r in sub)})
        for control in ['original','clipped_ray','line_joint','ray_no_history']:
            a=[r for r in rows if r['family']==family and r['method']=='typed_ray' and r['stage']=='final'];b=[lookup[(r['scene'],r['asset'],family,control,'original' if control=='original' else 'final')] for r in a];paired.append({'family':family,'method':'typed_ray','control':control,'stage':'final','gained':sum(x['success'] and not y['success'] for x,y in zip(a,b)),'lost':sum(y['success'] and not x['success'] for x,y in zip(a,b)),'ID_gained':sum(x['both_ids'] and not y['both_ids'] for x,y in zip(a,b)),'ID_lost':sum(y['both_ids'] and not x['both_ids'] for x,y in zip(a,b)),'common_correct_IDs':sum(x['both_ids'] and y['both_ids'] for x,y in zip(a,b)),'common_ID_success_gained':sum(x['both_ids'] and y['both_ids'] and x['success'] and not y['success'] for x,y in zip(a,b)),'common_ID_success_lost':sum(x['both_ids'] and y['both_ids'] and y['success'] and not x['success'] for x,y in zip(a,b))})
    assert [summary[f]['original']['original']['success'] for f in summary]==[115,80]
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'breakdown.json',breakdown);save(root/'evaluation_sources.json',{'evaluator_sha256':sha(Path(__file__)),'truth_sha256':truth_hash,'GT_only_after_full_prediction_seal':True,'GT_not_used_for_current_retuning':True,'primary_stage':'final','same_history_material_anchor_and_5cm_threshold':True});print(summary,flush=True)


if __name__=='__main__':main()
