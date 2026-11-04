"""Physical ID and actual historical material displacement, after full seal."""
from pathlib import Path
import pickle,sys
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(read('configs/uncertain_face_incidence_graph_v1.json')['output_root']);cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    methods=['original']+cfg['methods'];rows=[];graph_rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry'])
        for result in read(root/tag/'predictions.json'):
            i=result['asset'];ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i];true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0];ids=bool(result['selected_slots'][0]==truth['slot1'][i] and result['selected_slots'][1]==truth['slot2'][i])
            for method,pred in result['predictions'].items():
                positions=np.array([transform(anchor[None],np.array(pred[k]))[0] for k in ['T1','T2']]);error=float(np.linalg.norm(positions[1]-positions[0]-true_d));rows.append({'scene':tag,'geometry':run['geometry'],'asset':i,'family':result['family'],'method':method,'both_ids':ids,'error_m':error,'success':bool(ids and error<=.05)})
            for method in ['initial_graph']+cfg['methods']:
                face=result['graph']['faces'];offset=[f['offset'] for f in face]
                if method!='initial_graph' and method!='normal_fixed':offset=result['parameters'][method][12:12+len(face)]
                dist=[min(abs(np.array([box[j][f['axis']] for box in geometry[i] for j in [0,1]])-d)) for f,d in zip(face,offset)];graph_rows.append({'scene':tag,'asset':i,'family':result['family'],'method':method,'both_ids':ids,'faces':len(face),'edges':len(result['graph']['edges']),'face_distances_to_any_generator_plane_m':list(map(float,dist)),'postseal_optimistic_geometry_diagnostic_not_surface_precision':True})
        print('SCORED',tag,flush=True)
    lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows};summary={};paired=[]
    for family in ['catalogue_plane','material_shared']:
        summary[family]={}
        for method in methods:
            rr=[r for r in rows if r['family']==family and r['method']==method];summary[family][method]={'n':len(rr),'success':sum(r['success'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'correct_ID_median_error_m':float(np.median([r['error_m'] for r in rr if r['both_ids']]))}
            for control in ['original','normal_fixed','normal_joint','bounded_no_incidence']:
                if control==method:continue
                cc=[lookup[(r['scene'],r['asset'],family,control)] for r in rr];gain=sum(a['success'] and not b['success'] for a,b in zip(rr,cc));loss=sum(b['success'] and not a['success'] for a,b in zip(rr,cc));paired.append({'family':family,'method':method,'control':control,'gained':gain,'lost':loss,'net':gain-loss})
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'graph_geometry_postseal.json',graph_rows);save(root/'evaluation_sources.json',{'evaluator_sha256':sha(Path(__file__)),'truth_sha256':truth_hash,'GT_only_after_full_prediction_seal':True,'GT_not_used_for_current_retuning':True});print(summary,flush=True)


if __name__=='__main__':main()
