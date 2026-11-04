"""Post-seal full physical-material task scores and pre-existing failure panel."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box


def main():
    root=Path('results/material_domain_observation_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);prior=Path(cfg['plane_root'])
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    for p,h in read(root/'manifest.json')['sources'].items():assert sha(p)==h,p
    rows=[];scene_rows=[];source_checks=[]
    for run in read(root/'runs.json'):
        src=base/run['scene'];dest=root/run['scene'];pred=read(dest/'predictions.json');records=read(dest/'records.json');costs=np.load(dest/'costs.npz')
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        geom=geometry_for(run['seed'],cfg['objects'],run['geometry']);local=[]
        for i,(bb,a,ray) in enumerate(zip(inp['branches'],inp['anchors'],inp['historical_rays'])):
            correct=[next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1]]
            o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geom[i]);assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8
            material=o+z*r;A,B=truth['poses'][:,i];true_d=(B[:3,:3]-A[:3,:3])@material+B[:3,3]-A[:3,3]
            source_checks.append({'scene':run['scene'],'asset':i,'true_pair':correct,'finite_coverage':bool(np.isfinite(costs[f'coverage_{i}'][tuple(correct)])),
                'finite_independent':bool(np.isfinite(costs[f'independent_{i}'][tuple(correct)]))})
            branch_record={(v['epoch'],v['branch']):v for v in records[i]['branches']}
            for method,result in pred.items():
                chosen=next((v for v in result['selection'] if v[0]==i),None) if result['selection'] else None
                error=None;ids=False;matches=None;pose=False
                if chosen:
                    _,k,l=chosen;one,two=bb[0][k],bb[1][l];TA,TB=one['T'],two['T']
                    d=(TB[:3,:3]-TA[:3,:3])@a+TB[:3,3]-TA[:3,3];error=float(np.linalg.norm(d-true_d))
                    ids=bool(one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]);pose=[k,l]==correct
                    matches=[branch_record[(t,j)]['matches'] for t,j in enumerate([k,l])]
                row={'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':method,'resolved':chosen is not None,'status':result['status'],
                    'both_ids':ids,'physical_anchor_error_m':error,'true_pose_pair':pose,'identity_and_material_5cm':bool(ids and error is not None and error<=.05),
                    'positive_reference_ray_matches':matches,'one_or_more_epochs_without_positive_match':bool(matches is not None and min(matches)==0)}
                rows.append(row);local.append(row)
        scene_rows.append({'scene':run['scene'],'geometry':run['geometry'],'methods':{m:{'status':p['status'],'success':sum(v['identity_and_material_5cm'] for v in local if v['method']==m),
            'both_ids':sum(v['both_ids'] for v in local if v['method']==m),'resolved':sum(v['resolved'] for v in local if v['method']==m)} for m,p in pred.items()},
            'shared_equals_independent_selection':pred['shared_material_domain']['selection']==pred['independent_domains']['selection'],
            'shared_equals_coverage_selection':pred['shared_material_domain']['selection']==pred['coverage_only']['selection']})
    summary={};methods=['ordinary_nn','coverage_only','independent_domains','shared_material_domain']
    for mode in cfg['geometry_modes']:
        summary[mode]={}
        for method in methods:
            rr=[v for v in rows if v['geometry']==mode and v['method']==method]
            summary[mode][method]={'n':len(rr),'resolved':sum(v['resolved'] for v in rr),'both_ids':sum(v['both_ids'] for v in rr),
                'identity_and_material_5cm':sum(v['identity_and_material_5cm'] for v in rr),'zero_positive_reference_epoch':sum(v['one_or_more_epochs_without_positive_match'] for v in rr)}
    lookup={(v['scene'],v['asset'],v['method']):v for v in rows};paired=[]
    for mode in cfg['geometry_modes']+['all']:
        for control in ['ordinary_nn','coverage_only','independent_domains']:
            gained=lost=0
            for v in rows:
                if v['method']!='shared_material_domain' or (mode!='all' and v['geometry']!=mode):continue
                old=lookup[(v['scene'],v['asset'],control)]
                gained+=v['identity_and_material_5cm'] and not old['identity_and_material_5cm'];lost+=old['identity_and_material_5cm'] and not v['identity_and_material_5cm']
            paired.append({'geometry':mode,'shared_vs':control,'gained':int(gained),'lost':int(lost),'net':int(gained-lost)})
    old_failure=read(prior/'mechanism_diagnosis.json');panel=[]
    for old in old_failure['correct_identity_wrong_material']:
        panel.append({'scene':old['scene'],'asset':old['asset'],'previous_error_m':old['physical_error_m'],
                      'new_results':{m:lookup[(old['scene'],old['asset'],m)] for m in methods}})
    holes=[]
    for old in old_failure['bad_true_patches']:
        holes.append({'scene':old['scene'],'asset':old['asset'],'finite_true_pair':next(v for v in source_checks if v['scene']==old['scene'] and v['asset']==old['asset']),
                      'new_results':{m:lookup[(old['scene'],old['asset'],m)] for m in methods}})
    save(root/'scores.json',rows);save(root/'summary.json',summary);save(root/'scene_results.json',scene_rows);save(root/'paired_outcomes.json',paired)
    save(root/'true_pair_finiteness.json',source_checks);save(root/'previous_failure_panel.json',{'posthoc_panel_not_training':True,'quantification_15':panel,'concavity_conditions':holes})
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'truth_sources':{v['scene']:sha(base/v['scene']/'evaluation_only.pkl') for v in read(root/'runs.json')}})
    print(summary,flush=True);print('PAIRED',paired,flush=True)


if __name__=='__main__':main()
