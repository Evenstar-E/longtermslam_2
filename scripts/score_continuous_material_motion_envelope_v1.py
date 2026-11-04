"""Full physical-material evaluation of the ordinary motion-domain control."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box,transform


def main():
    root=Path('results/continuous_material_motion_envelope_v1');base=Path('results/continuous_material_pose_mainline_v1');cfg=read(base/'config.json');source=Path(cfg['base_root'])
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=[];rankings=[];old_rows=read(base/'scores.json');old_lookup={(r['scene'],r['asset'],r['method']):r for r in old_rows}
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (base/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (source/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (source/tag/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(dest/'predictions.json');allowed=np.load(dest/'allowed.npz');geometry=geometry_for(run['seed'],cfg['objects'],run['geometry'])
        selected={m:{v[0]:v[1:] for v in p['selection']} for m,p in pred.items()}
        for i,(bb,anchor,ray) in enumerate(zip(candidates,inp['anchors'],inp['historical_rays'])):
            o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);material=o+z*r
            A,B=truth['poses'][:,i];true_d=transform(material[None],B)[0]-transform(material[None],A)[0]
            positions=[np.array([transform(anchor[None],b['T'])[0] for b in branches]) for branches in bb]
            correct=[np.array([j for j,b in enumerate(branches) if b['slot']==truth[f'slot{t+1}'][i] and allowed[f'{i}_{t}'][j]]) for t,branches in enumerate(bb)]
            errors=np.linalg.norm(positions[1][correct[1]][None]-positions[0][correct[0]][:,None]-true_d,axis=-1)
            for m,result in pred.items():
                k,l=selected[m][i];one,two=bb[0][k],bb[1][l];error=float(np.linalg.norm(positions[1][l]-positions[0][k]-true_d));ids=bool(one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i])
                row={'scene':tag,'geometry':run['geometry'],'asset':i,'method':m,'resolved':True,'both_ids':ids,'physical_anchor_error_m':error,'identity_and_material_5cm':bool(ids and error<=.05),
                    'correct_ids_wrong_material':bool(ids and error>.05),'continuous_epochs':int(one['source']!='catalogue')+int(two['source']!='catalogue')};rows.append(row)
                if ids and error>.05 and m!='envelope_fixed_nn_ids_refined':
                    score=m[len('envelope_'):];costs=[np.array([branches[j]['scores'][score] for j in ix]) for branches,ix in zip(bb,correct)];grid=costs[0][:,None]+costs[1][None,:];good=errors<=.05
                    values=np.where(good,grid,np.inf);a,b=np.unravel_index(np.argmin(values),values.shape);selected_cost=one['scores'][score]+two['scores'][score]
                    assert np.isfinite(values[a,b]) and values[a,b]>=selected_cost-1e-8
                    rankings.append({'scene':tag,'geometry':run['geometry'],'asset':i,'method':m,'selected':[k,l],'selected_error_m':error,'selected_cost':selected_cost,
                        'best_good_pair':[int(correct[0][a]),int(correct[1][b])],'best_good_error_m':float(errors[a,b]),'best_good_cost':float(values[a,b]),'good_minus_selected_cost':float(values[a,b]-selected_cost),
                        'witness':'Even within metadata-derived motion envelope, retained correct-ID material pair has worse score than selected wrong-material pair.'})
    summary={};paired=[]
    for mode in cfg['geometry_modes']+['all']:
        summary[mode]={}
        for method in list(pred):
            rr=[r for r in rows if r['method']==method and (mode=='all' or r['geometry']==mode)]
            summary[mode][method]={field:sum(r[field] for r in rr) for field in ['resolved','both_ids','identity_and_material_5cm','correct_ids_wrong_material','continuous_epochs']};summary[mode][method]['n']=len(rr)
            for baseline in ['catalogue_nn','expanded_nn']:
                old=[old_lookup[(r['scene'],r['asset'],baseline)] for r in rr];gain=sum(a['identity_and_material_5cm'] and not b['identity_and_material_5cm'] for a,b in zip(rr,old));loss=sum(b['identity_and_material_5cm'] and not a['identity_and_material_5cm'] for a,b in zip(rr,old))
                paired.append({'geometry':mode,'method':method,'baseline':baseline,'gained':gain,'lost':loss,'net':gain-loss})
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'ranking_failures.json',rankings)
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'base_scores_sha256':sha(base/'scores.json'),'GT_only_after_seal':True,'truth_sources':{r['scene']:sha(source/r['scene']/'evaluation_only.pkl') for r in read(root/'runs.json')}})
    print(summary,flush=True);print('RANK_FAILURES',len(rankings),flush=True)


if __name__=='__main__':main()
