"""Post-seal physical-anchor evaluation and true-branch observation diagnostics."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box


def main():
    root=Path('results/first_return_material_surface_v1');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    manifest=read(root/'manifest.json')
    for p,h in manifest['sources'].items():assert sha(p)==h,p
    cfg=read(root/'config.json');rows=[];diagnosis=[];envelopes=[]
    for run in read(root/'runs.json'):
        dest=root/run['scene']
        with (dest/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        masks=np.load(dest/'feasible_masks.npz');records=read(dest/'records.json');pred=read(dest/'predictions.json')
        geometry=geometry_for(run['seed'],cfg['objects'],run['geometry'])
        checks={tuple(v['key']):v for v in read(dest/'visibility_checks.json')}
        for i,(bb,a,ray) in enumerate(zip(inp['branches'],inp['anchors'],inp['historical_rays'])):
            correct=[next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1]]
            origin=np.array(ray['origin']);r=np.array(ray['ray'])
            z=min(float(ray_box(origin,r,lo,hi)) for lo,hi in geometry[i]);assert abs(z-ray['depth'])<=cfg['depth_noise_bound_m']+1e-8
            material=origin+z*r;A,B=truth['poses'][:,i]
            true_d=(B[:3,:3]-A[:3,:3])@material+B[:3,3]-A[:3,3]
            for method,result in pred.items():
                selected=next((v for v in result['selection'] if v[0]==i),None) if result['selection'] else None
                error=None;identity=False
                if selected:
                    _,k,l=selected;one,two=bb[0][k],bb[1][l]
                    d=(two['T'][:3,:3]-one['T'][:3,:3])@a+two['T'][:3,3]-one['T'][:3,3]
                    error=float(np.linalg.norm(d-true_d));identity=bool(one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i])
                rows.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':method,'resolved':selected is not None,
                             'status':result['status'],'both_ids':identity,'physical_anchor_error_m':error,'identity_and_material_5cm':bool(identity and error is not None and error<=.05)})
            for method,surface in [('point_visibility',False),('first_return_surface',True)]:
                retained=bool(masks[f'{method}_{i}'][tuple(correct)])
                trace=[]
                one,two=bb[0][correct[0]],bb[1][correct[1]]
                for s,t,source,target in [(0,1,None,one),(0,2,None,two),(1,2,one,two)]:
                    key=(s,t,i if s==0 else source['slot'],target['slot'],-1 if s==0 else source['code'],target['code'],surface)
                    item=checks.get(key)
                    trace.append({'periods':[s,t],'tested':item is not None,'result':item})
                diagnosis.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':method,'true_pair_retained':retained,'true_checks':trace})
                env=records[i]['methods'][method]['material_envelope'];contained=bool(not env['empty'] and np.all(true_d>=np.array(env['lower'])-1e-9) and np.all(true_d<=np.array(env['upper'])+1e-9))
                midpoint_error=float(np.linalg.norm((np.array(env['lower'])+np.array(env['upper']))/2-true_d)) if not env['empty'] else None
                envelopes.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'method':method,'contains_truth':contained,
                                  'midpoint_error_m':midpoint_error,'wrong_report_5cm':bool(env['report'] and midpoint_error>.05),**env})
    summary={}
    for mode in cfg['geometry_modes']:
        summary[mode]={}
        for method in ['ordinary_nn','point_visibility','first_return_surface']:
            rr=[r for r in rows if r['geometry']==mode and r['method']==method];dd=[r for r in diagnosis if r['geometry']==mode and r['method']==method]
            ee=[r for r in envelopes if r['geometry']==mode and r['method']==method]
            summary[mode][method]={'n':len(rr),'resolved':sum(r['resolved'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),
                                  'identity_and_material_5cm':sum(r['identity_and_material_5cm'] for r in rr),
                                  'true_pair_retained':sum(r['true_pair_retained'] for r in dd) if dd else None,
                                  'material_envelopes_contain_truth':sum(r['contains_truth'] for r in ee) if ee else None,
                                  'reports':sum(r['report'] for r in ee) if ee else None,
                                  'reports_excluding_truth':sum(r['report'] and not r['contains_truth'] for r in ee) if ee else None,
                                  'wrong_reports_5cm':sum(r['wrong_report_5cm'] for r in ee) if ee else None}
    save(root/'scores.json',rows);save(root/'true_branch_diagnosis.json',diagnosis);save(root/'material_envelopes.json',envelopes);save(root/'summary.json',summary)
    save(root/'evaluation_sources.json',{'scorer_sha256':sha(Path(__file__)),'evaluation_only':{r['scene']:sha(root/r['scene']/'evaluation_only.pkl') for r in read(root/'runs.json')}})
    print(summary,flush=True)


if __name__=='__main__':main()
