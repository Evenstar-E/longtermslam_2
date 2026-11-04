"""Post-seal raw RGB residuals on near-same material, not a predictor."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,setup
from inspection_decisions.material_continuous_surface_correspondence_v14 import solve_offset
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));_,frames,ids,_,_,_,views,banks,_=setup(c,'rich');private_chart_material(scene,banks,ids);findex=[{id(f):h for h,f in enumerate(row)} for row in frames];rows=[];triples=[]
    @lru_cache(maxsize=16)
    def label(e,fi):
        with np.load(scene/records[int(ids[e][fi])]['labels']) as f:return f['material_xyz'],f['instance']
    for i in range(24):
        a=banks[0][i];res=[];matched=[]
        for e in [1,2]:
            slot=int(perm[e,i]);T=poses[e,i]@inverse(poses[0,i]);p=apply(T,a['xyz']);distance=np.full(len(p),np.inf);colors=np.zeros_like(p)
            for f in views[e][slot]:
                r=solve_offset(p,a['rgb'],f,slot,c,False);qgrid,idgrid=label(e,findex[e][id(f)]);uv=np.rint(r['uv']).astype(int);uv[:,0]=np.clip(uv[:,0],0,f['depth'].shape[1]-1);uv[:,1]=np.clip(uv[:,1],0,f['depth'].shape[0]-1);dm=np.linalg.norm(qgrid[uv[:,1],uv[:,0]]-a['q'],axis=1);valid=r['valid']&(idgrid[uv[:,1],uv[:,0]]==i)&(dm<=.003);take=valid&(dm<distance);colors[take]=f['rgb'][uv[take,1],uv[take,0]]/255.;distance[take]=dm[take]
            valid=np.isfinite(distance);r=colors-a['rgb'];res.append(r);matched.append(valid);rr=r[valid]
            rows.append(dict(asset=i,epoch=e,rear=int(perm[2,i]%3==0),material_radius_m=.003,matches=int(valid.sum()),rgb_l2_median=float(np.median(np.linalg.norm(rr,axis=1))) if len(rr) else None,rgb_channel_bias=rr.mean(0).tolist() if len(rr) else []))
        both=matched[0]&matched[1]
        if both.any():triples.append(np.c_[res[0][both],res[1][both]])
    X=np.concatenate(triples) if triples else np.empty((0,6));cov=np.cov(X.T).tolist() if len(X)>1 else None
    write_csv(root/'material_radiometry.csv',rows);save(root/'material_radiometry_summary.json',dict(variant='rich',material_radius_m=.003,paired_triple_points=len(X),residual_mean=X.mean(0).tolist() if len(X) else [],residual_covariance_6x6=cov,total_pair_queries=sum(r['matches'] for r in rows),median_asset_epoch_rgb_l2=float(np.median([r['rgb_l2_median'] for r in rows if r['matches']])),render_fact_source='inspection_decisions/textured_material_dataset_v1.py:199-218',render_fact='RGB=albedo*(.58+.42*max(n_world dot light_epoch,0))*exposure_frame*white_balance_frame + noise; raw RGB is not invariant albedo.',scope='GT true pose/identity and3mm nearest native material filter; near-point texture variation remains. Correlated points; no statistical significance/causal share claimed. Never used to fit or choose predictions. Shared fixed isotropic RGB covariance omits multiplicative albedo-dependent inter-channel covariance, light/pose means, and repaint null.'))
    print('RADIOMETRY',len(X),'triple points',sum(r['matches'] for r in rows),'pair points')


if __name__=='__main__':main()
