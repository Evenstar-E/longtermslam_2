"""Raw-input rebuilding of the full 192 predictions, without evaluator data."""
from pathlib import Path
import pickle
import numpy as np
import torch
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from inspection_decisions.surface_measure_first_return_v1 import (ray_data,build_dictionary,history_posterior,unknown_exposure,candidate_stats,cost)
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector


def main():
    torch.set_num_threads(1);root=Path('results/surface_measure_first_return_v1');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8
    checks=0;max_error=0.;gradient_fits=0;surface_counts=[];unknown_history=[]
    for name,h in seal['sha256'].items():assert sha(root/name)==h;checks+=1
    manifest=read(root/'manifest.json')
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];src=root/run['storage'];cfg=read(src/'scene_config.json');base=Path(cfg['base_root'])
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,colors,_=load_frames(base,tag,centers,cfg);data=[ray_data(ff,cfg) for ff in frames];predictions=read(src/'predictions.json');z=np.load(src/'costs.npz');C=z['C'];appearance=z['appearance'];limits=manifest['motion_limits'];assignments=read(src/'assignments.json')
        for mi,m in enumerate(cfg['methods']):
            slots=np.array(assignments[m]['slots']);assert all(sorted(row)==list(range(24)) for row in slots)
            assert abs(C[mi,np.arange(24),slots[0],slots[1]].sum()-assignments[m]['log']['energy'])<1e-6;checks+=3
        for i in range(24):
            model=build_dictionary(frames[0],inp['T0'][i],i,cfg);a,b,meta=history_posterior(data[0],inp['T0'][i],i,model,cfg)
            old_model=read(src/f'material_{i:02d}.json')['model'];old=np.load(src/f'posterior_{i:02d}.npz');assert np.max(abs(a-old['a']))<1e-9 and np.max(abs(b-old['b']))<1e-9;checks+=2
            for key in ['center','normal','precision','surface_mass']:
                assert np.allclose(model[key],np.array(old_model[key]).reshape(model[key].shape),rtol=0,atol=1e-12);checks+=1
            surface_counts.append(dict(scene=tag,asset=i,history_views=run['history_views'],surfaces=len(model['center']),history_hits=meta['history_hits'],
                                      history_surface_hit_fraction=float((a[model['V']:]-model['surface_mass']).sum()/max(meta['history_hits'],1)),
                                      mean_footprint_m=float(np.mean(model['footprint'])) if len(model['footprint']) else None))
            # History no-positive cells still have strictly nonzero optical prior.
            assert (a[:model['V']]>0).all() and (b>0).all();checks+=2
            for mi,m in enumerate(cfg['methods']):
                row=predictions[i]['methods'][m];l,r=row['candidate_indices'];slots=np.array(row['stages']['candidate']['slots']);S=[]
                for t,ci in enumerate([l,r]):
                    pool=pools[i][t];assert len(pool)==288 and set(p['slot'] for p in pool)==set(range(24));checks+=2
                    candidate=pool[ci];assert candidate['slot']==slots[t] and np.array_equal(candidate['T'],row['stages']['candidate'][f'T{t+1}']);checks+=1
                    S.append(candidate_stats(data[t+1],np.array(candidate['T']),slots[t],model,a,b,cfg))
                initial=cost(a,b,*S,m);error=abs(initial-row['log']['initial_cost']);max_error=max(max_error,error);assert error<1e-6;checks+=1
                j,k=slots;error=abs(initial+appearance[i,0,l]+appearance[i,1,r]-C[mi,i,j,k]);max_error=max(max_error,error);assert error<1e-6;checks+=1
                final=[]
                for t in range(2):
                    T=np.array(row['stages']['final'][f'T{t+1}']);x=pose_vector(T,centers[slots[t]]);assert np.linalg.norm(x[:3])<=limits[0]+1e-10 and np.linalg.norm(x[3:])<=limits[1]+1e-10;checks+=2
                    final.append(candidate_stats(data[t+1],T,slots[t],model,a,b,cfg))
                value=cost(a,b,*final,m);error=abs(value-row['log']['final_cost']);max_error=max(max_error,error);assert error<1e-6;checks+=1
                assert value<=initial+1e-7 and np.max(np.diff(row['log']['cost_trace']),initial=0)<=1e-8;checks+=2;gradient_fits+=1
        print('AUDITED',tag,checks,'error',max_error,flush=True)
    assert gradient_fits==768
    save(root/'audit.json',dict(passed=True,GT_read=False,checks=checks,continuous_fits_rebuilt=gradient_fits,max_raw_cost_error=max_error,models=surface_counts,all_288_retained=True,
         inference_reduced_pair_scalar_checks=192*4*24*24,scope='All selected initial and final predictions rebuilt from raw RGB-D. Runner rebuilt every reduced ID-pair scalar. Audit does not independently regenerate every discarded candidate pair. Exposure pixel quadrature is shared approximation, not full-pixel likelihood.'))
    print('AUDIT_PASS',checks,flush=True)


if __name__=='__main__':main()
