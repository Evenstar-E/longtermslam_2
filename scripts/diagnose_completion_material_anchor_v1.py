"""Postseal physical reference-point check; generation geometry is evaluator-only.

Inference used a noisy historical reference coordinate. Recover its source ray
from measured data, then intersect that ray with generation geometry only to
evaluate the actual material reference. Does not update old predictions.
"""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.completion_catalog_scene_v1 import shapes,ray_box,inverse,transform

def main():
    root=Path('results/completion_compatible_anchor_v1');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json')
    for p,v in seal['sha256'].items():assert sha(root/p)==v
    rows=[];sources=[]
    for run in read(root/'runs.json'):
        folder=root/run['scene'];hv=int(run['scene'].rsplit('h',1)[1])
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        with (folder/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(folder/'predictions.json');geometry=shapes(cfg['objects'])
        for i,(bb,a) in enumerate(zip(p['branches'],p['anchors'])):
            T0=next(b['T'] for b in bb[0] if b['slot']==i and b['code']==0)
            source=None
            for view in range(hv):
                frame=np.load(folder/f'frame_0_{view}.npz');C=frame['C'];cam=inverse(transform(a[None],T0),C)[0]
                pixel=np.rint(cam[:2]/cam[2]*cfg['focal_px']+np.array([(cfg['width']-1)/2,(cfg['height']-1)/2])).astype(int)
                u,v=pixel
                if 0<=u<cfg['width'] and 0<=v<cfg['height']:
                    depth=float(frame['depth'][v,u]);ray_cam=np.array([(u-(cfg['width']-1)/2)/cfg['focal_px'],(v-(cfg['height']-1)/2)/cfg['focal_px'],1.])
                    measured=inverse(transform((ray_cam*depth)[None],C),T0)[0]
                    if np.linalg.norm(measured-a)<1e-8:
                        origin=inverse(C[:3,3][None],T0)[0];ray=ray_cam@C[:3,:3].T@T0[:3,:3]
                        source=(view,int(u),int(v),origin,ray,depth);break
            assert source is not None
            view,u,v,o,r,depth=source
            true_depth=min(float(ray_box(o,r,lo,hi)) for lo,hi in geometry[i])
            assert np.isfinite(true_depth) and abs(true_depth-depth)<=cfg['depth_noise_bound_m']+1e-8
            material=o+true_depth*r
            R1,R2=truth['poses'][0,i,:3,:3],truth['poses'][1,i,:3,:3]
            d=R2@material+truth['poses'][1,i,:3,3]-R1@material-truth['poses'][0,i,:3,3]
            bias=float(np.linalg.norm(d-truth['anchor_displacement'][i]))
            # Source-depth uncertainty is shared by the two epochs, not two
            # independent anchor errors. Candidate-wise added axis halfwidth:
            # epsilon * abs((R2-R1) @ ray).
            sources.append({'scene':run['scene'],'asset':i,'view':view,'pixel':[u,v],
                            'measured_anchor':a.tolist(),'physical_material_anchor':material.tolist(),
                            'canonical_ray':r.tolist(),'depth_error_m':depth-true_depth,
                            'original_vs_physical_displacement_m':bias})
            for method,result in pred.items():
                _,k,l=next(v for v in result['selection'] if v[0]==i)
                one,two=bb[0][k],bb[1][l]
                estimate=two['T'][:3,:3]@a+two['T'][:3,3]-one['T'][:3,:3]@a-one['T'][:3,3]
                identity=one['slot']==truth['slot1'][i] and two['slot']==truth['slot2'][i]
                error=float(np.linalg.norm(estimate-d))
                rows.append({'scene':run['scene'],'asset':i,'method':method,'identity_and_physical_anchor_5cm':bool(identity and error<=.05),'physical_anchor_error_m':error})
    summary={m:sum(r['identity_and_physical_anchor_5cm'] for r in rows if r['method']==m) for m in ['plain_nn','pairwise','full_shared']}
    summary['n']=len(sources);summary['max_original_vs_physical_displacement_m']=max(r['original_vs_physical_displacement_m'] for r in sources)
    save(root/'material_anchor_diagnosis.json',{'GT_assisted':True,'updates_predictions':False,'sources':sources,'rows':rows,'summary':summary,
        'limit':'Original envelope concerns measured reference coordinate; no physical-anchor enclosure claimed without including shared source-depth uncertainty.'})
    print(summary)

if __name__=='__main__':main()
