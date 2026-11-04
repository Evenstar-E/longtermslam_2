"""GT-selected8rear pairs, all public seeds; diagnosis NEVER task success."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from scripts.run_material_surface_distribution_v15b import configuration,load
from scripts.prepare_material_surface_fields_v15 import load_fields
from scripts.run_material_continuous_surface_correspondence_v14c import setup,dense_records
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from inspection_decisions.material_surface_distribution_v15b import dense_modes,compatibility
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);torch.set_flush_denormal(True);c=configuration();root=Path(c['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];ic,frames,ids,_,_,_,views,banks,_=setup(c,'rich');dense=dense_records(views);load_fields(c,'rich',banks,dense);scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));perm=np.array(truth['permutations']);poses=np.array(truth['poses']);anchors=material_anchors(scene);private_chart_material(scene,banks,ids);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));index={id(f):h for h,f in enumerate(frames[2])};rows=[];payload={}
    @lru_cache(maxsize=16)
    def label(k):
        with np.load(scene/records[int(ids[2][k])]['labels']) as f:return f['material_xyz'],f['instance']
    for i in range(24):
        slot=int(perm[2,i])
        if slot%3:continue
        a=banks[0][i];b=dense[2][slot];m=dense_modes(a,b,c);tree=b['_tree'];_,j=tree.query(m['point']);targetq=np.zeros_like(b['xyz']);targetid=np.full(len(targetq),-1,int)
        for fi,f in enumerate(views[2][slot]):
            h=b['frame']==fi;uv=b['uv'][h].astype(int);qgrid,idgrid=label(index[id(f)]);targetq[h]=qgrid[uv[:,1],uv[:,0]];targetid[h]=idgrid[uv[:,1],uv[:,0]]
        true=(targetid[j]==a['physical_id'][m['source']])&(np.linalg.norm(targetq[j]-a['q'][m['source']],axis=1)<=.015);x=a['xyz'][m['source']];anchor=apply(poses[0,i],anchors[i]);reference=apply(poses[2,i],anchors[i])
        for regional in [False,True]:
            C,y=compatibility(a,m,c,regional);C2=(C@C)*C;v=np.sqrt(np.maximum(m['mass'],1e-12)).astype(np.float32)
            for _ in range(8):v=C2@v;v/=max(np.linalg.norm(v),1e-12)
            order=np.argsort(-v);rank=np.empty(len(v),int);rank[order]=np.arange(len(v));original=[]
            for h in order:
                if original and np.min(np.linalg.norm(x[original]-x[h],axis=1))<.04:continue
                original.append(int(h))
                if len(original)>=12:break
            Tlist=[];seeds=[]
            for h in range(len(v)):
                weights=C2[h]*v;chosen=[];seen=set()
                for k in np.argsort(-weights):
                    source=int(m['source'][k])
                    if source in seen:continue
                    seen.add(source);chosen.append(int(k))
                    if len(chosen)>=24:break
                ix=np.array(chosen);T=rigid(x[ix],y[ix],weights[ix]+1e-12)
                if T is not None:Tlist.append(T);seeds.append(h)
            U=np.array(Tlist);E=np.quantile(np.linalg.norm(np.einsum('bij,nj->bni',U[:,:3,:3],anchor)+U[:,None,:3,3]-reference[None],axis=-1),.95,axis=1);best=int(E.argmin());seed=seeds[best];original_error=[E[z] for z,h in enumerate(seeds) if h in original]
            rows.append(dict(asset=i,regional=int(regional),all_seeds=len(v),true_peak_edges=int(true.sum()),original_seeds=len(original),original_true_peak_seeds=int(true[original].sum()),best_true_peak_spectral_rank=int(rank[true].min()) if true.any() else None,original_best_pair_error_m=float(min(original_error)),allseed_best_pair_error_m=float(E[best]),allseed_qualified_poses=int((E<=.05).sum()),allseed_best_seed_spectral_rank=int(rank[seed]),allseed_best_seed_true_peak=int(true[seed])))
            payload[f'{i}_{int(regional)}_T']=U;payload[f'{i}_{int(regional)}_seed']=np.array(seeds);payload[f'{i}_{int(regional)}_errors']=E
    write_csv(root/'seed_failure_probe.csv',rows);np.savez_compressed(root/'seed_failure_probe.npz',**payload);summary=[]
    for regional in [0,1]:
        rr=[r for r in rows if r['regional']==regional];summary.append(dict(regional=regional,original_pair_qualified=sum(r['original_best_pair_error_m']<=.05 for r in rr),allseed_pair_qualified=sum(r['allseed_best_pair_error_m']<=.05 for r in rr),original_true_peak_seed_total=sum(r['original_true_peak_seeds'] for r in rr),allseed_qualified_poses=sum(r['allseed_qualified_poses'] for r in rr)))
    save(root/'seed_failure_probe_summary.json',dict(summary=summary,scope='GT chooses8true H2rear identity pairs; every candidate seed otherwise processed using frozen public C2/global spectral weights andsame24source-unique consensus. Extra1536seed search budget, selected/reported by GT pair5cm after seal. NOT prediction improvement, complete task success, equal-budget algorithm or independent confirmation. Tests final12seed truncation separately from within-consensus global weighting.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
