"""Postseal interventions on all 31 changed task outcomes, NEVER inference.

Same frozen pose pool; oracle source-face roles vs oracle physical line-support
links diagnose independent per-ray alternative selection. No new algorithm win.
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.diagnose_set_valued_material_edges_v1 import domain
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions,translate_regions
from inspection_decisions.sparse_ray_material_union_cache_v1 import prepare_reference,nearest_support_costs


def source_axes(obs,x,boxes):
    R=Rotation.from_rotvec(x[:3]).as_matrix();o=(obs['origin']-x[3:])@R;r=obs['ray_inside']@R;z=min(float(ray_box(o,r,lo,hi)) for lo,hi in boxes);q=o+z*r;axes=[]
    for a in range(3):
        if abs(r[a])<1e-10:continue
        na=-np.sign(r[a])*np.eye(3)[a]
        if min(abs(np.unique(np.array(boxes)[:,:,a])-q[a]))<1e-7 and domain(q,na,boxes):axes.append(a)
    return set(axes)


def true_track_regions(regions,events):
    result=[]
    for row in regions:
        for event in events.get(row['ray_id'],[]):
            if row['source_axis']!=event['source_axis'] or row['exit_axis']!=event['exit_axis']:continue
            token=tuple(event['key'])+tuple(np.round(event['support_coordinates_m'],9));obj=dict(row);obj['key']=token;result.append(obj)
    return result


def main():
    root=Path('results/sparse_ray_material_incidence_v1_run4');cfg=read(root/'config.json');base=Path(cfg['base_root']);assert read(root/'prediction_seal.json')['complete'];scores=read(root/'scores.json');lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in scores};geometry_rows=read(root/'physical_geometry_gate.json')['rows'];event_map={}
    for event in geometry_rows:event_map.setdefault((event['scene'],event['epoch'],event['slot']),{}).setdefault(event['ray_id'],[]).append(event)
    panel=[];aggregate={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (root/tag/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        slots=inp['T0'][:,:3,3];geometry=geometry_for(run['seed'],24,run['geometry'])
        for pred in read(root/tag/'predictions.json'):
            i=pred['asset'];family=pred['family'];old=lookup[(tag,i,family,'original')];new=lookup[(tag,i,family,'sparse_unordered')]
            if old['identity_and_material_5cm']==new['identity_and_material_5cm']:continue
            assert old['both_ids'] and new['both_ids'];kind='gained' if new['identity_and_material_5cm'] else 'lost';T1=np.array(pred['initial_T1']);T2=np.array(pred['initial_T2']);selected=pred['selected_slots'];xx=[np.zeros(6),pose_vector(T1,slots[selected[0]]),pose_vector(T2,slots[selected[1]])];rows=[raw[0][i],raw[1][selected[0]],raw[2][selected[1]]];slot_list=[i,*selected];estimated=[];axis_maps=[]
            for t,(obs,x,slot) in enumerate(zip(rows,xx,slot_list)):
                regions,_=all_regions(x,obs,cfg);estimated.append(regions);trueT=inp['T0'][i] if t==0 else truth['poses'][t-1,i];truex=pose_vector(trueT,slots[slot]);axis_maps.append({r['ray_id']:source_axes(r,truex,geometry[i]) for r in obs})
            face_reference=prepare_reference([h for t in [0,1] for h in estimated[t] if h['source_axis'] in axis_maps[t][h['ray_id']]])
            track_reference=prepare_reference([h for t in [0,1] for h in true_track_regions(estimated[t],event_map.get((tag,t,slot_list[t]),{}))]);late_events=event_map.get((tag,2,slot_list[2]),{});scan=[]
            for s in pred['scan']:
                current=translate_regions(estimated[2],s['epsilon_m']);face=[h for h in current if h['source_axis'] in axis_maps[2][h['ray_id']]];track=true_track_regions(current,late_events);cf,_=nearest_support_costs(face,face_reference,len(rows[2]),cfg);ct,_=nearest_support_costs(track,track_reference,len(rows[2]),cfg);scan.append({'epsilon_m':s['epsilon_m'],'unrestricted':s['sparse_unordered'],'oracle_source_face':cf['mean_capped_squared_cost'],'oracle_physical_edge_support_links':ct['mean_capped_squared_cost']})
            ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;true_d=transform(material[None],truth['poses'][1,i])[0]-transform(material[None],truth['poses'][0,i])[0];a=inp['anchors'][i];initial_error=transform(a[None],T2)[0]-transform(a[None],T1)[0]-true_d;v=T2[:3,2];methods={}
            for method in ['unrestricted','oracle_source_face','oracle_physical_edge_support_links']:
                minimum=min(s[method] for s in scan);index=min([j for j,s in enumerate(scan) if s[method]<=minimum+cfg['selection_tolerance']],key=lambda j:abs(scan[j]['epsilon_m']));eps=scan[index]['epsilon_m'];error=float(np.linalg.norm(initial_error+eps*v));methods[method]={'epsilon_m':eps,'error_m':error,'material_5cm':error<=.05,'score':scan[index][method]}
            panel.append({'scene':tag,'asset':i,'family':family,'change_vs_original':kind,'initial_error_m':old['physical_anchor_error_m'],'methods':methods,'scan':scan,'GT_derived_labels_used_for_diagnosis_only':True})
        print('ASSOCIATION_PANEL',tag,flush=True)
    for family in ['catalogue_plane','material_shared']:
        aggregate[family]={}
        for kind in ['gained','lost']:
            rr=[r for r in panel if r['family']==family and r['change_vs_original']==kind];aggregate[family][kind]={'n':len(rr),'oracle_source_face_material_success':sum(r['methods']['oracle_source_face']['material_5cm'] for r in rr),'oracle_physical_edge_support_links_material_success':sum(r['methods']['oracle_physical_edge_support_links']['material_5cm'] for r in rr),'unrestricted_material_success':sum(r['methods']['unrestricted']['material_5cm'] for r in rr)}
    save(root/'association_interventions.json',{'summary':aggregate,'rows':panel,'source_sha256':sha(Path(__file__)),'postseal_diagnostic_not_new_task_performance':True,'panel_all_changed_task_outcomes':True,'GT_labels_never_fed_back_into_inference':True});print(aggregate,flush=True)


if __name__=='__main__':main()
