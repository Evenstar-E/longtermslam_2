"""No-GT audit of sealed raw-ray statistics, chosen factors and final costs."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_first_return_gamma_material_v4 import load_frames
from inspection_decisions.first_return_gamma_material_v1 import stats,scalar_cost


def main():
    root=Path('results/first_return_gamma_material_v4');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8;manifest=read(root/'manifest.json');checks=0;error=0.;rawerror=0.;descent=0;fits=0;rays=0;gaps=[]
    for p,h in seal['sha256'].items():assert sha(root/p)==h;checks+=1
    for field in ['sources','inputs']:
        for p,h in manifest[field].items():assert sha(p)==h;checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;cfg=read(dest/'scene_config.json');limits=manifest['motion_limits'];rows=read(dest/'predictions.json');assign=read(dest/'assignments.json');cert=read(dest/'workspace_certificate.json');assert cert['extent_m']>cert['observed_max_radius_m']+limits[1];checks+=1
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (Path(cfg['parent_root'])/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,colors,inventory=load_frames(Path(cfg['base_root']),tag,centers,cfg);assert inventory==read(dest/'raw_ray_inventory.json');rays+=sum(r['unique_pixels'] for r in inventory);checks+=1
        H=np.load(dest/'history_stats.npz')['H'];current=[np.load(dest/f'current_stats_{t}.npz') for t in range(2)];z=np.load(dest/'costs.npz');C=z['C'];choices=z['choices'];rgb=z['appearance'];Q=[v['Q'] for v in current];mapping=[v['mapping'] for v in current]
        assert C.shape==(4,24,24,24) and np.isfinite(C).all();checks+=1
        # Full H statistics are regenerated; random continuous candidate and
        # every selected final pose are independently rebuilt from raw frames.
        for i in range(24):
            h,_=stats(frames[0],inp['T0'][i],i,cfg);d=float(np.max(abs(h-H[i])));rawerror=max(rawerror,d);assert d<1e-11;checks+=1
            for t in range(2):
                assert len(pools[i][t])==288 and {b['slot'] for b in pools[i][t]}==set(range(24));checks+=1
                k=(i*53+17*t)%288;b=pools[i][t][k];q,_=stats(frames[t+1],np.array(b['T']),b['slot'],cfg);d=float(np.max(abs(q-Q[t][mapping[t][i,k]])));rawerror=max(rawerror,d);assert d<1e-11;checks+=1
            for mi,m in enumerate(cfg['methods']):
                # Full 24x24 selected minima are checked as individual exact
                # joint factors. This does not certify all discarded minima.
                for j in range(24):
                    for k in range(24):
                        aa,bb=choices[mi,i,j,k];assert pools[i][0][aa]['slot']==j and pools[i][1][bb]['slot']==k
                        a=Q[0][mapping[0][i,aa]];b=Q[1][mapping[1][i,bb]];v=scalar_cost(H[i],a,b,cfg,m)+rgb[i,0,aa]+rgb[i,1,bb];d=abs(v-C[mi,i,j,k]);error=max(error,d);assert d<1e-7;checks+=1
                v=rows[i]['methods'][m];j,k=v['stages']['final']['slots'];aa,bb=v['candidate_indices'];assert [aa,bb]==choices[mi,i,j,k].tolist();assert np.array_equal(np.array(assign[m]['slots'])[:,i],[j,k]);checks+=2
                final=v['stages']['final'];qq=[stats(frames[t+1],np.array(final['T'+str(t+1)]),[j,k][t],cfg)[0] for t in range(2)];cost=scalar_cost(H[i],*qq,cfg,m);d=abs(cost-v['log']['final_cost']);error=max(error,d);assert d<1e-7;checks+=1
                trace=np.array(v['log']['cost_trace']);assert np.all(np.diff(trace)<0) and trace[-1]<=trace[0]+1e-8;checks+=1;descent+=int(len(trace)>1);fits+=1
                p=np.array(v['final_parameters'])
                for t in range(2):
                    assert np.linalg.norm(p[t*6:t*6+3])<=limits[0]+1e-10 and np.linalg.norm(p[t*6+3:t*6+6])<=limits[1]+1e-10;checks+=1
        for mi,m in enumerate(cfg['methods']):
            slots=np.array(assign[m]['slots']);assert all(sorted(x)==list(range(24)) for x in slots);value=float(C[mi,np.arange(24),slots[0],slots[1]].sum());assert abs(value-assign[m]['log']['energy'])<1e-7;gaps.append({'scene':tag,'method':m,'raw_LP_gap':assign[m]['log']['gap'],'numeric_only_not_formal_certificate':True});checks+=2
        print('AUDIT',tag,'checks',checks,'max_error',error,flush=True)
    save(root/'audit.json',{'passed':True,'checks':checks,'max_factor_or_final_cost_error':error,'max_raw_statistic_error':rawerror,'continuous_fits':fits,'fits_with_descent':descent,'unique_raw_rays_across_conditions':rays,'not_independent_samples':True,'GT_read':False,'discarded_candidate_optimality_recomputed':False,'all_selected_slotpair_factors_reconstructed':True,'auditor_sha256':sha(Path(__file__)),'LP_gaps':gaps})


if __name__=='__main__':main()
