"""Truth accessed only after complete all-task inference seal verification."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/shared_material_incidence_v1_run2');cfg=read(root/'config.json');base=Path(cfg['base_root']);seal=read(root/'prediction_seal.json');assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    methods=['original','nn','plane','sparse_ordered','sparse_unordered','graph_independent','graph_shared','graph_shared_no_domain'];rows=[];truth_hash={};graphs=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry'])
        for result in read(root/tag/'predictions.json'):
            i=result['asset'];ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i];true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0];ids=bool(result['selected_slots'][0]==truth['slot1'][i] and result['selected_slots'][1]==truth['slot2'][i]);zero=min(result['scan'],key=lambda s:abs(s['epsilon_m']))
            for method,pred in result['predictions'].items():
                positions=np.array([transform(anchor[None],np.array(pred[k]))[0] for k in ['T1','T2']]);err=float(np.linalg.norm(positions[1]-positions[0]-true_d));scan=result['scan'];values=[s[method] for s in scan] if method!='original' else None;selected=scan[pred['scan_index']]
                rows.append({'scene':tag,'geometry':run['geometry'],'asset':i,'family':result['family'],'method':method,'both_ids':ids,'error_m':err,'success':bool(ids and err<=.05),'epsilon_m':pred['epsilon_m'],'flat':bool(max(values)-min(values)<1e-10) if values else None,'selected_contradictory_pixels':selected['graph_diagnostics']['contradictory_pixels'],'selected_unmatched_rays':selected['graph_diagnostics']['unmatched_rays'],'zero_contradictory_pixels':zero['graph_diagnostics']['contradictory_pixels'],'pixels':selected['graph_diagnostics']['pixels'],'late_rays':result['raw_ray_counts'][2],**result['graph_counts']})
        for entry in read(root/tag/'graphs.json'):
            i=entry['asset'];F=entry['graph']['faces'];dist=[]
            for face in F:
                a=face['axis'];offsets=np.array([box[j][a] for box in geometry[i] for j in [0,1]]);dist.append(float(np.min(abs(offsets-face['offset']))))
            graphs.append({'scene':tag,'asset':i,'family':entry['family'],'face_count':len(F),'edge_count':len(entry['graph']['edges']),'faces_within_6mm_of_any_generator_plane':sum(d<=.006 for d in dist),'face_nearest_generator_plane_distances_m':dist,'postseal_upper_bound_precision_not_visible_surface_truth':True})
        print('SCORED',tag,flush=True)
    lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows};summary={};paired=[];retention=[]
    for family in ['catalogue_plane','material_shared']:
        summary[family]={}
        for method in methods:
            rr=[r for r in rows if r['family']==family and r['method']==method];summary[family][method]={'n':len(rr),'success':sum(r['success'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'moved':sum(abs(r['epsilon_m'])>1e-12 for r in rr),'flat':sum(r['flat'] is True for r in rr)}
            for control in ['original','sparse_unordered','graph_independent']:
                if control==method:continue
                cc=[lookup[(r['scene'],r['asset'],family,control)] for r in rr];gain=sum(a['success'] and not b['success'] for a,b in zip(rr,cc));loss=sum(b['success'] and not a['success'] for a,b in zip(rr,cc));paired.append({'family':family,'method':method,'control':control,'gained':gain,'lost':loss,'net':gain-loss})
        rr=[r for r in rows if r['family']==family and r['method']=='sparse_unordered'];gains=[r for r in rr if r['success'] and not lookup[(r['scene'],r['asset'],family,'original')]['success']];losses=[r for r in rr if not r['success'] and lookup[(r['scene'],r['asset'],family,'original')]['success']]
        for method in ['graph_independent','graph_shared','graph_shared_no_domain']:
            retention.append({'family':family,'method':method,'old_gain_panel_n':len(gains),'old_gains_preserved':sum(lookup[(r['scene'],r['asset'],family,method)]['success'] for r in gains),'old_loss_panel_n':len(losses),'old_losses_repaired':sum(lookup[(r['scene'],r['asset'],family,method)]['success'] for r in losses)})
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'retention.json',retention);save(root/'graph_diagnostics_postseal.json',graphs);save(root/'evaluation_sources.json',{'evaluator_sha256':sha(Path(__file__)),'truth_sha256':truth_hash,'GT_only_after_complete_seal':True,'GT_not_used_for_retuning':True});print(summary,flush=True);print('RETENTION',retention,flush=True)


if __name__=='__main__':main()
