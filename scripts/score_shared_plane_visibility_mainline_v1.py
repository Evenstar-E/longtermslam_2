"""Post-seal full-task evaluation; truth only appears in this evaluator."""
from pathlib import Path
import pickle
from collections import Counter
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box


def main():
    root=Path('results/shared_plane_visibility_mainline_v1');cfg=read(root/'config.json');base=Path(cfg['base_root'])
    seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    for p,h in read(root/'manifest.json')['sources'].items():assert sha(p)==h,p
    scores=[];diagnostics=[];env_rows=[];scene_results=[]
    methods=['nominal_plane','independent_period_planes','shared_period_plane']
    for run in read(root/'runs.json'):
        dest=root/run['scene'];src=base/run['scene']
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(dest/'predictions.json');records=read(dest/'records.json');masks=np.load(dest/'feasible_masks.npz')
        geometry=geometry_for(run['seed'],cfg['objects'],run['geometry']);local=[]
        for i,(bb,a,ray) in enumerate(zip(inp['branches'],inp['anchors'],inp['historical_rays'])):
            correct=[next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1]]
            o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i])
            assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8
            material=o+z*r;A,B=truth['poses'][:,i];true_d=(B[:3,:3]-A[:3,:3])@material+B[:3,3]-A[:3,3]
            rec=records[i];diag={'scene':run['scene'],'geometry':run['geometry'],'asset':i,'correct_pair':correct,'retained':{},'true_unary_failures':{}}
            with (dest/f'asset_{i:02d}_factors.pkl').open('rb') as f:factors=pickle.load(f)
            for t,k in enumerate(correct):
                diag['true_unary_failures'][str(t)]=[j for j,out in enumerate(factors['unary_trace'][t][k]) if out['feasible'] is False]
            for m in methods:
                kept=bool(masks[f'{m}_{i}'][tuple(correct)]);diag['retained'][m]=kept
                env=rec['envelopes'][m]
                contains=bool(not env['empty'] and np.all(true_d>=np.array(env['lower'])-1e-9) and np.all(true_d<=np.array(env['upper'])+1e-9))
                error=float(np.linalg.norm((np.array(env['lower'])+np.array(env['upper']))/2-true_d)) if not env['empty'] else None
                env_rows.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':m,'contains_truth':contains,
                    'midpoint_error_m':error,'wrong_report_5cm':bool(env['report'] and error>.05),'true_pair_retained':kept,**env})
            diagnostics.append(diag)
            for method,result in pred.items():
                chosen=next((v for v in result['selection'] if v[0]==i),None) if result['selection'] else None
                error=None;identity=False;branch_correct=False
                if chosen:
                    _,k,l=chosen;one,two=bb[0][k],bb[1][l];TA,TB=one['T'],two['T']
                    estimate=(TB[:3,:3]-TA[:3,:3])@a+TB[:3,3]-TA[:3,3];error=float(np.linalg.norm(estimate-true_d))
                    identity=bool(one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]);branch_correct=[k,l]==correct
                row={'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':method,'resolved':chosen is not None,'status':result['status'],
                     'both_ids':identity,'true_pose_pair':bool(branch_correct),'physical_anchor_error_m':error,'identity_and_material_5cm':bool(identity and error is not None and error<=.05)}
                scores.append(row);local.append(row)
        scene_results.append({'scene':run['scene'],'results':{m:{'success':sum(r['identity_and_material_5cm'] for r in local if r['method']==m),
                  'true_pairs_retained':sum(d['retained'][m] for d in diagnostics if d['scene']==run['scene']) if m in methods else None,
                  'status':p['status']} for m,p in pred.items()},
                  'same_selected_branches_independent_vs_shared':pred['independent_period_planes']['selection']==pred['shared_period_plane']['selection'],
                  'same_selected_branches_nominal_vs_shared':pred['nominal_plane']['selection']==pred['shared_period_plane']['selection']})
    summary={}
    for mode in cfg['geometry_modes']:
        summary[mode]={}
        for method in ['ordinary_nn']+methods:
            rr=[r for r in scores if r['geometry']==mode and r['method']==method];dd=[d for d in diagnostics if d['geometry']==mode];ee=[e for e in env_rows if e['geometry']==mode and e['method']==method]
            summary[mode][method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),
               'identity_and_material_5cm':sum(r['identity_and_material_5cm'] for r in rr),'true_pose_pair':sum(r['true_pose_pair'] for r in rr),
               'true_pair_retained':sum(d['retained'][method] for d in dd) if method in methods else None,
               'local_reports':sum(e['report'] for e in ee) if ee else None,'wrong_local_reports':sum(e['wrong_report_5cm'] for e in ee) if ee else None,
               'local_envelopes_contain_truth':sum(e['contains_truth'] for e in ee) if ee else None}
    paired=[]
    by_key={(r['scene'],r['asset'],r['method']):r for r in scores}
    for baseline in ['ordinary_nn','nominal_plane','independent_period_planes']:
        gained=lost=changed=0
        for r in scores:
            if r['method']!='shared_period_plane':continue
            b=by_key[(r['scene'],r['asset'],baseline)]
            gained+=r['identity_and_material_5cm'] and not b['identity_and_material_5cm'];lost+=b['identity_and_material_5cm'] and not r['identity_and_material_5cm']
        paired.append({'shared_vs':baseline,'gained':int(gained),'lost':int(lost),'net':int(gained-lost)})
    save(root/'scores.json',scores);save(root/'summary.json',summary);save(root/'true_branch_diagnosis.json',diagnostics)
    save(root/'material_envelopes.json',env_rows);save(root/'scene_results.json',scene_results);save(root/'paired_outcomes.json',paired)
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'truth_sources':{r['scene']:sha(base/r['scene']/'evaluation_only.pkl') for r in read(root/'runs.json')}})
    print(summary,flush=True);print('PAIRED',paired,flush=True)


if __name__=='__main__':main()
