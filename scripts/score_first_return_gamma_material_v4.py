"""Formal 192-asset material/identity task, only after ALL inference and audit."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/first_return_gamma_material_v4');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8 and read(root/'audit.json')['passed']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(src/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry']);new=read(root/tag/'predictions.json');previous=read(Path(cfg['parent_root'])/tag/'predictions.json');gibbs=read(Path('results/sparse_material_evidence_v1')/tag/'predictions.json')
        for i in range(24):
            ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i];true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0];predictions=[('hazard',m,stage,pred) for m,v in new[i]['methods'].items() for stage,pred in v['stages'].items()]
            predictions.append(('Gibbs','previous_joint_ray','final',gibbs[i]['methods']['joint_ray']['stages']['final']))
            for p in [r for r in previous if r['asset']==i]:
                predictions.append((p['family'],'original','final',p['original']))
                for m in ['typed_ray','line_joint']:predictions.append((p['family'],'previous_'+m,'final',p['methods'][m]['stages']['final']))
            for family,method,stage,pred in predictions:
                ids=bool(pred['slots'][0]==truth['slot1'][i] and pred['slots'][1]==truth['slot2'][i]);positions=np.array([transform(anchor[None],np.array(pred[k]))[0] for k in ['T1','T2']]);error=float(np.linalg.norm(positions[1]-positions[0]-true_d));rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'family':family,'method':method,'stage':stage,'slots':pred['slots'],'both_ids':ids,'error_m':error,'success':bool(ids and error<=.05)})
        print('SCORED',tag,flush=True)
    summary={};breakdown=[]
    for family in ['hazard','catalogue_plane','material_shared','Gibbs']:
        summary[family]={}
        for method in (cfg['methods'] if family=='hazard' else (['previous_joint_ray'] if family=='Gibbs' else ['original','previous_typed_ray','previous_line_joint'])):
            summary[family][method]={}
            for stage in (['candidate','final'] if family=='hazard' else ['final']):
                rr=[r for r in rows if (r['family'],r['method'],r['stage'])==(family,method,stage)];correct=[r['error_m'] for r in rr if r['both_ids']];summary[family][method][stage]={'n':len(rr),'success':sum(r['success'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'correct_ID_median_error_m':float(np.median(correct)) if correct else None}
                for geometry in ['aligned','offgrid']:
                    for h in [1,3]:
                        sub=[r for r in rr if r['geometry']==geometry and r['history_views']==h];breakdown.append({'family':family,'method':method,'stage':stage,'geometry':geometry,'history_views':h,'n':len(sub),'success':sum(r['success'] for r in sub),'both_ids':sum(r['both_ids'] for r in sub)})
    lookup={(r['scene'],r['asset'],r['family'],r['method'],r['stage']):r for r in rows};paired=[];main=[r for r in rows if (r['family'],r['method'],r['stage'])==('hazard','gamma_joint','final')]
    for family,method in [('hazard',m) for m in cfg['methods'] if m!='gamma_joint']+[(f,m) for f in ['catalogue_plane','material_shared'] for m in ['original','previous_typed_ray','previous_line_joint']]+[('Gibbs','previous_joint_ray')]:
        control=[lookup[(r['scene'],r['asset'],family,method,'final')] for r in main];paired.append({'method':'gamma_joint','control_family':family,'control':method,'gained':sum(a['success'] and not b['success'] for a,b in zip(main,control)),'lost':sum(b['success'] and not a['success'] for a,b in zip(main,control)),'ID_gained':sum(a['both_ids'] and not b['both_ids'] for a,b in zip(main,control)),'ID_lost':sum(b['both_ids'] and not a['both_ids'] for a,b in zip(main,control)),'common_correct_ID_tasks':sum(a['both_ids'] and b['both_ids'] for a,b in zip(main,control)),'common_correct_ID_success_gained':sum(a['both_ids'] and b['both_ids'] and a['success'] and not b['success'] for a,b in zip(main,control)),'common_correct_ID_success_lost':sum(a['both_ids'] and b['both_ids'] and b['success'] and not a['success'] for a,b in zip(main,control))})
    assert [summary[f]['original']['final']['success'] for f in ['catalogue_plane','material_shared']]==[115,80]
    assert [summary[f]['previous_typed_ray']['final']['success'] for f in ['catalogue_plane','material_shared']]==[120,83]
    assert summary['Gibbs']['previous_joint_ray']['final']['success']==85
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'breakdown.json',breakdown);save(root/'evaluation_sources.json',{'script_sha256':sha(Path(__file__)),'truth_sha256':truth_hash,'GT_read_only_after_full_seal_and_audit':True,'GT_not_used_to_retune_current_model':True,'primary_stage':'final','new_predictions_192_not384_independent_initializations':True});print('NEW_FINAL', {m:summary['hazard'][m]['final'] for m in cfg['methods']},flush=True)


if __name__=='__main__':main()
