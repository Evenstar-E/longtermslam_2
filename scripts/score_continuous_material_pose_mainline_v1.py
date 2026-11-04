"""Full-task physical-anchor scores, after the ENTIRE new inference is sealed."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box,transform


def main():
    root=Path('results/continuous_material_pose_mainline_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);prior=Path(cfg['plane_root'])
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=[];rankings=[];refinements=[];oracles=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:all_candidates=pickle.load(f)
        pred=read(dest/'predictions.json');logs=read(dest/'optimization_logs.json');geometry=geometry_for(run['seed'],cfg['objects'],run['geometry'])
        selected={m:{v[0]:v[1:] for v in p['selection']} if p['selection'] else {} for m,p in pred.items()}
        for i,(bb,anchor,ray) in enumerate(zip(all_candidates,inp['anchors'],inp['historical_rays'])):
            o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i]);assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8
            material=o+z*r;true_positions=np.array([transform(material[None],T)[0] for T in truth['poses'][:,i]]);true_d=true_positions[1]-true_positions[0]
            positions=[np.array([transform(anchor[None],b['T'])[0] for b in branches]) for branches in bb]
            correct_indices=[np.array([j for j,b in enumerate(branches) if b['slot']==truth[f'slot{t+1}'][i]]) for t,branches in enumerate(bb)]
            delta=positions[1][correct_indices[1]][None]-positions[0][correct_indices[0]][:,None]
            error_matrix=np.linalg.norm(delta-true_d,axis=-1)
            for expanded,name in [(False,'catalogue'),(True,'expanded')]:
                use=[np.array([j for j,k in enumerate(ii) if expanded or bb[t][k]['source']=='catalogue']) for t,ii in enumerate(correct_indices)]
                err=error_matrix[np.ix_(*use)];oracles.append({'scene':tag,'geometry':run['geometry'],'asset':i,'pool':name,'min_physical_error_m':float(err.min()),'good_correct_id_pairs':int((err<=.05).sum())})
            for method,result in pred.items():
                chosen=selected[method].get(i);error=None;ids=False;positive=None;continuous=0
                if chosen is not None:
                    k,l=chosen;one,two=bb[0][k],bb[1][l];estimate=positions[1][l]-positions[0][k];error=float(np.linalg.norm(estimate-true_d))
                    ids=bool(one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]);continuous=int(one['source']!='catalogue')+int(two['source']!='catalogue')
                    positive=[[b['support'][d]['positive'] for d in ['current_to_history','history_to_current']] for b in [one,two]]
                    if ids and error>.05 and (method.startswith('catalogue_') or method.startswith('expanded_')):
                        expanded=method.startswith('expanded_');score=method.split('_',1)[1]
                        costs=[np.array([b['scores'][score] for b in branches])[ix] for branches,ix in zip(bb,correct_indices)]
                        grid=costs[0][:,None]+costs[1][None,:];good=error_matrix<=.05
                        if not expanded:
                            old0=np.array([bb[0][j]['source']=='catalogue' for j in correct_indices[0]]);old1=np.array([bb[1][j]['source']=='catalogue' for j in correct_indices[1]])
                            good&=old0[:,None]&old1[None,:]
                        good_grid=np.where(good,grid,np.inf);a,b=np.unravel_index(np.argmin(good_grid),good_grid.shape)
                        candidate=[int(correct_indices[0][a]),int(correct_indices[1][b])];chosen_cost=one['scores'][score]+two['scores'][score];gap=float(good_grid[a,b]-chosen_cost)
                        assert np.isfinite(good_grid[a,b]) and gap>=-1e-8
                        rankings.append({'scene':tag,'geometry':run['geometry'],'asset':i,'method':method,'selected':chosen,'selected_error_m':error,
                            'best_good_correct_id_pair':candidate,'best_good_error_m':float(error_matrix[a,b]),'selected_cost':chosen_cost,'best_good_cost':float(good_grid[a,b]),'good_minus_selected_cost':gap,
                            'witness':'Correct-ID material solution exists in same pool but this separable score selects a worse material pose.'})
                rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'asset':i,'method':method,'resolved':chosen is not None,'status':result['status'],
                    'both_ids':ids,'physical_anchor_error_m':error,'identity_and_material_5cm':bool(ids and error is not None and error<=.05),'correct_ids_wrong_material':bool(ids and error is not None and error>.05),
                    'continuous_epochs':continuous,'positive_counts':positive,'zero_positive_direction':bool(positive is not None and min(v for pp in positive for v in pp)==0)})
            for log in [v for v in logs if v['asset']==i]:
                t=log['epoch'];old=bb[t][log['parent']];j=log['endpoint_index'];new=bb[t][j] if j is not None else None
                correct_seed=bool(old['slot']==truth[f'slot{t+1}'][i] and old['code']==truth[f'code{t+1}'][i])
                if not correct_seed:continue
                initial_error=float(np.linalg.norm(positions[t][log['parent']]-true_positions[t]));final_error=float(np.linalg.norm(positions[t][j]-true_positions[t])) if j is not None else None
                refinements.append({'scene':tag,'geometry':run['geometry'],'asset':i,'epoch':t,'initial_is_generated_true_pose':True,'initial_error_m':initial_error,
                    'complete':new is not None,'endpoint_error_m':final_error,'pushed_beyond_5cm':bool(final_error is not None and final_error>.05),
                    'nn_cost_before':old['scores']['nn'],'nn_cost_after':new['scores']['nn'] if new is not None else None,
                    'plane_cost_before':old['scores']['plane'],'plane_cost_after':new['scores']['plane'] if new is not None else None})
        print('SCORED',tag,flush=True)
    methods=list(pred);summary={};paired=[];lookup={(r['scene'],r['asset'],r['method']):r for r in rows}
    for mode in cfg['geometry_modes']+['all']:
        summary[mode]={}
        for method in methods:
            rr=[r for r in rows if r['method']==method and (mode=='all' or r['geometry']==mode)]
            summary[mode][method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'identity_and_material_5cm':sum(r['identity_and_material_5cm'] for r in rr),
                'correct_ids_wrong_material':sum(r['correct_ids_wrong_material'] for r in rr),'continuous_epochs_selected':sum(r['continuous_epochs'] for r in rr),'zero_positive_direction':sum(r['zero_positive_direction'] for r in rr)}
            if method=='catalogue_nn':continue
            rr0=[lookup[(r['scene'],r['asset'],'catalogue_nn')] for r in rr]
            gained=sum(a['identity_and_material_5cm'] and not b['identity_and_material_5cm'] for a,b in zip(rr,rr0));lost=sum(b['identity_and_material_5cm'] and not a['identity_and_material_5cm'] for a,b in zip(rr,rr0))
            paired.append({'geometry':mode,'method':method,'baseline':'catalogue_nn','gained':gained,'lost':lost,'net':gained-lost})
    old_summary=read(prior/'summary.json')
    for mode in cfg['geometry_modes']:
        for field in ['resolved','both_ids','identity_and_material_5cm']:
            assert summary[mode]['catalogue_nn'][field]==old_summary[mode]['ordinary_nn'][field]
            assert summary[mode]['sealed_independent_planes'][field]==old_summary[mode]['independent_period_planes'][field]
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'paired_outcomes.json',paired);save(root/'ranking_failures.json',rankings);save(root/'true_pose_refinement.json',refinements);save(root/'oracle.json',oracles)
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'truth_sources':{r['scene']:sha(base/r['scene']/'evaluation_only.pkl') for r in read(root/'runs.json')},'truth_used_only_after_seal':True})
    print('SUMMARY',summary,flush=True);print('TRUTH_SEEDS',len(refinements),'PUSHED',sum(r['pushed_beyond_5cm'] for r in refinements),'FAILED',sum(not r['complete'] for r in refinements),flush=True)


if __name__=='__main__':main()
