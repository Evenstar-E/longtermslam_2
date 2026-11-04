"""Postseal all common endpoints: own-fit scores and oracle capacity ONLY."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/uncertain_face_incidence_graph_v2');cfg=read(root/'config.json');assert read(root/'prediction_seal.json')['complete'];parent=Path(cfg['parent_root']);records=[];own=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        geometry=geometry_for(run['seed'],24,run['geometry'])
        for row,old in zip(read(root/tag/'predictions.json'),read(parent/tag/'predictions.json')):
            i=row['asset'];ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i];d=transform(material[None],truth['poses'][1,i])[0]-transform(material[None],truth['poses'][0,i])[0];s1,s2=row['selected_slots'];ids=bool(s1==truth['slot1'][i] and s2==truth['slot2'][i])
            def error(pred):return float(np.linalg.norm(transform(anchor[None],np.array(pred['T2']))[0]-transform(anchor[None],np.array(pred['T1']))[0]-d))
            pool=[('original',row['predictions']['original'])]+[('v1_'+m,old['predictions'][m]) for m in cfg['methods']]+[('v2_'+m,row['own_fit_predictions'][m]) for m in cfg['methods']];ee=[(name,error(pred)) for name,pred in pool];record={'scene':tag,'asset':i,'family':row['family'],'both_ids':ids,'endpoint_errors_m':dict(ee),'oracle_success':bool(ids and min(e for _,e in ee)<=.05),'original_success':bool(ids and ee[0][1]<=.05)};records.append(record)
            for method,pred in row['own_fit_predictions'].items():
                err=error(pred);own.append({'scene':tag,'asset':i,'family':row['family'],'method':method,'error_m':err,'success':bool(ids and err<=.05),'selected_from_common_pool_success':bool(ids and error(row['predictions'][method])<=.05)})
    summary={}
    for family in ['catalogue_plane','material_shared']:
        rr=[r for r in records if r['family']==family];summary[family]={'oracle9_endpoint_success':sum(r['oracle_success'] for r in rr),'original_success':sum(r['original_success'] for r in rr),'own_fit_success':{m:sum(r['success'] for r in own if r['family']==family and r['method']==m) for m in cfg['methods']}}
    save(root/'endpoint_capacity_postseal.json',{'summary':summary,'rows':records,'own_fit_rows':own,'oracle_only_after_full_seal_never_inference':True,'not_algorithm_accuracy':True});print(summary)


if __name__=='__main__':main()
