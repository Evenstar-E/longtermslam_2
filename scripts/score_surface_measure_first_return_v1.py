"""Uniform evaluator only after all 192 tasks sealed and no-GT audit passed."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/surface_measure_first_return_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['assets']==192 and read(root/'audit.json')['passed']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    for field in ['sources','inputs']:
        for p,h in read(root/'manifest.json')[field].items():assert sha(p)==h
    rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];base=Path(cfg['base_root'])/tag
        with (base/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (base/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        truth_hash[tag]=sha(base/'evaluation_only.pkl');geometry=geometry_for(run['seed'],24,run['geometry'])
        new=read(root/run['storage']/'predictions.json');voxel=read(Path(cfg['previous_root'])/tag/'predictions.json');strong=read(Path(cfg['parent_root'])/tag/'predictions.json')
        controlroot=Path('results/surface_measure_matched_voxel_v1');assert read(controlroot/'audit.json')['passed'];cr=next(r for r in read(controlroot/'runs.json') if r['scene']==tag);matched=read(controlroot/cr['storage']/'predictions.json')
        for i in range(24):
            ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r;anchor=inp['anchors'][i]
            positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=positions[1]-positions[0]
            predictions=[('surface',m,s,p) for m,v in new[i]['methods'].items() for s,p in v['stages'].items()]
            predictions += [('voxel',m,'final',v['stages']['final']) for m,v in voxel[i]['methods'].items()]
            predictions.append(('matched_voxel','shared','final',matched[i]['stages']['final']))
            for old in [p for p in strong if p['asset']==i]:
                predictions.append((old['family'],'original','final',old['original']))
                predictions += [(old['family'],m,'final',old['methods'][m]['stages']['final']) for m in ['typed_ray','line_joint']]
            for family,m,stage,pred in predictions:
                ids=bool(pred['slots'][0]==truth['slot1'][i] and pred['slots'][1]==truth['slot2'][i]);p=np.array([transform(anchor[None],np.array(pred[k]))[0] for k in ['T1','T2']]);error=float(np.linalg.norm(p[1]-p[0]-true_d))
                rows.append(dict(scene=tag,geometry=run['geometry'],history_views=run['history_views'],asset=i,family=family,method=m,stage=stage,slots=pred['slots'],both_ids=ids,error_m=error,success=bool(ids and error<=.05)))
        print('SCORED',tag,flush=True)
    keys=sorted({(r['family'],r['method'],r['stage']) for r in rows});summary=[]
    for family,m,stage in keys:
        for h in [None,1,3]:
            rr=[r for r in rows if (r['family'],r['method'],r['stage'])==(family,m,stage) and (h is None or r['history_views']==h)]
            summary.append(dict(family=family,method=m,stage=stage,history_views=h,n=len(rr),success=sum(r['success'] for r in rr),both_ids=sum(r['both_ids'] for r in rr),correct_ID_median_error_m=float(np.median([r['error_m'] for r in rr if r['both_ids']])) if any(r['both_ids'] for r in rr) else None))
    lookup={(r['scene'],r['asset'],r['family'],r['method'],r['stage']):r for r in rows};main=[r for r in rows if (r['family'],r['method'],r['stage'])==('surface','surface_shared','final')];paired=[]
    controls=[('surface',m) for m in cfg['methods'] if m!='surface_shared']+[('matched_voxel','shared'),('voxel','gamma_joint'),('catalogue_plane','typed_ray'),('catalogue_plane','line_joint')]
    for family,m in controls:
        for h in [None,1,3]:
            aa=[r for r in main if h is None or r['history_views']==h];bb=[lookup[(r['scene'],r['asset'],family,m,'final')] for r in aa]
            paired.append(dict(control_family=family,control=m,history_views=h,n=len(aa),gained=sum(a['success'] and not b['success'] for a,b in zip(aa,bb)),lost=sum(b['success'] and not a['success'] for a,b in zip(aa,bb)),common_correct_ID_tasks=sum(a['both_ids'] and b['both_ids'] for a,b in zip(aa,bb)),common_ID_quant_gained=sum(a['both_ids'] and b['both_ids'] and a['success'] and not b['success'] for a,b in zip(aa,bb)),common_ID_quant_lost=sum(a['both_ids'] and b['both_ids'] and b['success'] and not a['success'] for a,b in zip(aa,bb))))
    get=lambda f,m,h:next(r['success'] for r in summary if (r['family'],r['method'],r['stage'],r['history_views'])==(f,m,'final',h))
    assert get('voxel','gamma_joint',None)==93 and get('catalogue_plane','typed_ray',None)==120
    assert get('voxel','gamma_joint',3)==52 and get('catalogue_plane','typed_ray',3)==74
    panel=[]
    for a in [r for r in main if r['history_views']==3]:
        old=lookup[(a['scene'],a['asset'],'voxel','gamma_joint','final')];strong=lookup[(a['scene'],a['asset'],'catalogue_plane','typed_ray','final')]
        panel.append(dict(scene=a['scene'],asset=a['asset'],old_success=old['success'],strong_success=strong['success'],new_success=a['success'],old_strong_only=bool(strong['success'] and not old['success']),recovered_strong_only=bool(strong['success'] and not old['success'] and a['success']),new_loss_vs_old=bool(old['success'] and not a['success'])))
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'H3_recovery_panel.json',panel)
    save(root/'evaluation_sources.json',dict(script_sha256=sha(Path(__file__)),truth_sha256=truth_hash,GT_read_only_after_full_seal_and_audit=True,GT_not_used_to_retune=True,primary_stage='final',familiar_8_conditions_not_192_independent_scenes=True))
    print('FINAL_ALL192',[(r['method'],r['success'],r['both_ids']) for r in summary if r['family']=='surface' and r['stage']=='final' and r['history_views'] is None],flush=True)


if __name__=='__main__':main()
