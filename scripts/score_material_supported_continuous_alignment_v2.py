"""All 192 physical-material tasks; annotations only after full inference seal."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/material_supported_continuous_alignment_v2');cfg=read(root/'config.json');base=Path(cfg['base_root']);baseline=Path(cfg['baseline_root'])
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=[];old=read(baseline/'scores.json');lookup={(r['scene'],r['asset'],r['method']):r for r in old};drift=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        pred=read(dest/'predictions.json');geometry=geometry_for(run['seed'],24,run['geometry'])
        selections={m:{i:[k,l] for i,k,l in p['selection']} if p['selection'] else {} for m,p in pred.items()}
        for i,(bb,a,ray) in enumerate(zip(candidates,inp['anchors'],inp['historical_rays'])):
            o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8
            material=o+z*r;true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0]
            for method,result in pred.items():
                pair=selections[method].get(i);error=None;ids=False;epoch_errors=None;sources=None
                if pair is not None:
                    chosen=[bb[t][k] for t,k in enumerate(pair)];positions=np.array([transform(a[None],b['T'])[0] for b in chosen]);error=float(np.linalg.norm(positions[1]-positions[0]-true_d));epoch_errors=np.linalg.norm(positions-true_positions,axis=1).tolist();sources=[b['source'] for b in chosen]
                    ids=bool(chosen[0]['slot']==truth['slot1'][i] and chosen[1]['slot']==truth['slot2'][i])
                rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'method':method,'status':result['status'],'resolved':pair is not None,
                    'both_ids':ids,'physical_anchor_error_m':error,'epoch_anchor_errors_m':epoch_errors,'identity_and_material_5cm':bool(ids and error is not None and error<=.05),
                    'correct_ids_wrong_material':bool(ids and error is not None and error>.05),'selected_pair':pair,'sources':sources})
            for t in [0,1]:
                k=next(j for j,b in enumerate(bb[t][:54]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i])
                for j,b in enumerate(bb[t]):
                    if b['source']=='bounded_icp' and b['parent']==k:
                        e=float(np.linalg.norm(transform(a[None],b['T'])[0]-true_positions[t]));drift.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'epoch':t,'initial_true_seed':k,'endpoint':j,'anchor_error_m':e,'pushed_beyond_5cm':e>.05})
        print('SCORED',tag,flush=True)
    methods=list(pred);summary={};paired=[];new_lookup={(r['scene'],r['asset'],r['method']):r for r in rows}
    for mode in ['aligned','offgrid','all']:
        summary[mode]={}
        for method in methods:
            rr=[r for r in rows if r['method']==method and (mode=='all' or r['geometry']==mode)]
            summary[mode][method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'optimal_global_tasks':sum(r['status']==0 for r in rr),'both_ids':sum(r['both_ids'] for r in rr),
                'identity_and_material_5cm':sum(r['identity_and_material_5cm'] for r in rr),'correct_ids_wrong_material':sum(r['correct_ids_wrong_material'] for r in rr)}
            for control in ['catalogue_nn','catalogue_plane','bounded_icp_plane','material_independent']:
                if method==control:continue
                previous=[(lookup if control.startswith('catalogue_') else new_lookup)[(r['scene'],r['asset'],control)] for r in rr]
                gain=sum(a['identity_and_material_5cm'] and not b['identity_and_material_5cm'] for a,b in zip(rr,previous));loss=sum(b['identity_and_material_5cm'] and not a['identity_and_material_5cm'] for a,b in zip(rr,previous))
                paired.append({'geometry':mode,'method':method,'control':control,'gained':gain,'lost':loss,'net':gain-loss})
    same=[]
    for run in read(root/'runs.json'):
        p=read(root/run['scene']/'predictions.json');same.append({'scene':run['scene'],'shared_same_independent':p['material_shared']['selection']==p['material_independent']['selection'],
            'catalogue_shared_same_independent':p['catalogue_material_shared']['selection']==p['catalogue_material_independent']['selection']})
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'selected_comparisons.json',same);save(root/'true_seed_bounded_drift.json',drift)
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'baseline_scores_sha256':sha(baseline/'scores.json'),'GT_read_only_after_complete_seal':True,
        'truth_sources':{r['scene']:sha(base/r['scene']/'evaluation_only.pkl') for r in read(root/'runs.json')}})
    print(summary,flush=True);print('SHARED_PAIRED',[p for p in paired if p['geometry']=='all' and p['method']=='material_shared'],flush=True)


if __name__=='__main__':main()
