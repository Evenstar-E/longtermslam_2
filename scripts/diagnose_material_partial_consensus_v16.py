"""Post-seal material attribution; GT-selected pairs NEVER predictions."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from scripts.run_material_partial_consensus_v16 import configuration,load,FAMILIES
from scripts.prepare_material_surface_fields_v15 import load_fields
from scripts.run_material_continuous_surface_correspondence_v14c import setup,dense_records
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha
from inspection_decisions.material_surface_distribution_v15b import dense_modes
from inspection_decisions.material_partial_consensus_graph_v16 import graph_state,neighborhood,refine,partial_statistics,apply


def main():
    torch.set_num_threads(1);torch.set_flush_denormal(True);c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read']
    for p,h in seal['sources'].items():assert sha(p)==h,p
    ic,frames,ids,_,_,_,views,banks,_=setup(c,'rich');dense=dense_records(views);load_fields(c,'rich',banks,dense);scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);private_chart_material(scene,banks,ids);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));raw=load(root/'rich_pair_proposals.npz');rows=[];seedrows=[];maximum=0.;checks=0
    @lru_cache(maxsize=32)
    def label(e,k):
        with np.load(scene/records[int(ids[e][k])]['labels']) as z:return z['material_xyz'],z['instance']
    index={id(f):h for h,f in enumerate(frames[2])}
    for i,q in enumerate(anchors):
        slot=int(perm[2,i]);rear=slot%3==0;x=apply(poses[0,i],q);y=apply(poses[2,i],q)
        def err(T):return float(np.quantile(np.linalg.norm(apply(T,x)-y,axis=1),.95))
        prefix=f'1_{i}_{slot}';row=dict(asset=i,rear=int(rear),pair_visited=int(prefix+'_global24_tls_raw' in raw),global_true_peak_seeds=0,local_true_peak_seeds=0,true_peak_edges=None,true_peak_mass_median=None,false_peak_mass_median=None,target_volume_m3=None)
        for f in FAMILIES:
            for stage in ['raw','selected']:
                U=raw.get(prefix+'_'+f+('_raw' if stage=='raw' else ''));E=np.array([err(T) for T in U]) if U is not None else np.array([]);row[f+'_'+stage+'_error_m']=float(E.min()) if len(E) else None;row[f+'_'+stage+'_qualified']=int(len(E)>0 and E.min()<=.05)
        rows.append(row)
        if not rear or not row['pair_visited']:continue
        a=banks[0][i];b=dense[2][slot];m=dense_modes(a,b,c);_,j=b['_tree'].query(m['point']);targetq=np.zeros_like(b['xyz']);targetid=np.full(len(targetq),-1,int)
        for fi,f in enumerate(views[2][slot]):
            h=b['frame']==fi;uv=b['uv'][h].astype(int);qgrid,idgrid=label(2,index[id(f)]);targetq[h]=qgrid[uv[:,1],uv[:,0]];targetid[h]=idgrid[uv[:,1],uv[:,0]]
        true=(targetid[j]==a['physical_id'][m['source']])&(np.linalg.norm(targetq[j]-a['q'][m['source']],axis=1)<=.015);g=graph_state(a,m,c);volume=float(np.prod(np.ptp(b['xyz'],axis=0)+c['outlier_padding_m']))
        row['global_true_peak_seeds']=int(true[raw[prefix+'_f_global_seeds']].sum());row['local_true_peak_seeds']=int(true[raw[prefix+'_f_local_seeds']].sum());row['true_peak_edges']=int(true.sum());row['true_peak_mass_median']=float(np.median(m['mass'][true])) if true.any() else None;row['false_peak_mass_median']=float(np.median(m['mass'][~true]));row['target_volume_m3']=volume
        for n,h in enumerate(raw[prefix+'_f_local_seeds']):
            key=prefix+f'_f_local3_tls_{n}_initial'
            if key not in raw:continue
            initial=raw[key];ix,w=neighborhood(g,m,int(h),c,True);TLS,path=refine(initial.copy(),g,m,ix,c,False,volume);partial,ppath=refine(initial.copy(),g,m,ix,c,True,volume)
            for f,T,p in [('local3_tls',TLS,path),('local3_partial',partial,ppath)]:
                assert np.max(abs(p-raw[prefix+f'_f_{f}_{n}_path']))<1e-8;assert min(np.max(abs(T-Q)) for Q in raw[prefix+'_'+f+'_raw'])<1e-8;checks+=2
            yy=g['y'][ix];_,target=np.unique(yy,axis=0,return_inverse=True);loss,weights,resp,si=partial_statistics(partial,g['x'][ix],yy,m['source'][ix],target,m['mass'][ix],c,volume);rowmax=float(np.max(np.bincount(si,weights=weights)));targetmax=float(np.max(np.bincount(target,weights=weights)));assert rowmax<=1+1e-8 and targetmax<=1+1e-8;checks+=1
            # At the true rigid pose: distinguish low feature prior from solver failure.
            truth_T=poses[2,i]@np.linalg.inv(poses[0,i]);_,gw,gr,_=partial_statistics(truth_T,g['x'][ix],yy,m['source'][ix],target,m['mass'][ix],c,volume)
            seedrows.append(dict(asset=i,seed=n,true_peak_seed=int(true[h]),true_pool_sources=len(np.unique(m['source'][ix[true[ix]]])),pool_sources=len(np.unique(m['source'][ix])),initial_error_m=err(initial),tls_error_m=err(TLS),partial_error_m=err(partial),partial_objective_drop=float(ppath[0]-ppath[-1]),true_weight_fraction=float(weights[true[ix]].sum()/max(weights.sum(),1e-30)),true_weight_fraction_at_gt_pose=float(gw[true[ix]].sum()/max(gw.sum(),1e-30)),true_responsibility_at_gt_pose_mean=float(gr[true[ix]].mean()) if true[ix].any() else None,capacity_active_targets=int((np.bincount(target,weights=resp)>1+1e-8).sum()),capacity_removed_mass_fraction=float((resp.sum()-weights.sum())/max(resp.sum(),1e-30)),source_capacity_max=rowmax,target_capacity_max=targetmax))
    write_csv(root/'rich_pair_diagnosis.csv',rows);write_csv(root/'rich_local_seed_diagnosis.csv',seedrows);rear=[r for r in rows if r['rear']];summary={}
    for f in FAMILIES:
        for stage in ['raw','selected']:summary[f+'_'+stage+'_qualified']=sum(r[f+'_'+stage+'_qualified'] for r in rear)
    save(root/'diagnosis_summary.json',dict(checks=checks,rear_pair_summary=summary,global_true_peak_seeds=sum(r.get('global_true_peak_seeds',0) for r in rear),local_true_peak_seeds=sum(r.get('local_true_peak_seeds',0) for r in rear),local_seed_count=len(seedrows),local_pools_at_least3_true_sources=sum(r['true_pool_sources']>=3 for r in seedrows),initial_qualified=sum(r['initial_error_m']<=.05 for r in seedrows),tls_qualified=sum(r['tls_error_m']<=.05 for r in seedrows),partial_qualified=sum(r['partial_error_m']<=.05 for r in seedrows),partial_gain_vs_tls=sum(r['partial_error_m']<=.05<r['tls_error_m'] for r in seedrows),partial_loss_vs_tls=sum(r['tls_error_m']<=.05<r['partial_error_m'] for r in seedrows),partial_lost_good_start=sum(r['initial_error_m']<=.05<r['partial_error_m'] for r in seedrows),capacity_active_seeds=sum(r['capacity_active_targets']>0 for r in seedrows),objective_descent_material_worse=sum(r['partial_objective_drop']>1e-10 and r['partial_error_m']>r['initial_error_m']+1e-8 for r in seedrows),scope='POST-SEAL GT-selected rich H2 true-identity rear8 pairs and local forward seeds. Pair<=5cm, not complete three-segment success or independent confirmation. True weights are diagnostics only, never inference.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
