"""Post-seal actual512 preservation and true-pose continuous-query support."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,setup
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from inspection_decisions.material_continuous_exposure_v14c import directed_match
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);s=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert s['complete']
    for p,h in s['sources'].items():assert sha(p)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));rows=[];edge_rows=[]
    for v in c['variants']:
        _,frames,ids,_,_,_,views,banks,_=setup(c,v);private_chart_material(scene,banks,ids);findex=[{id(f):h for h,f in enumerate(row)} for row in frames]
        @lru_cache(maxsize=16)
        def label(e,fi):
            with np.load(scene/records[int(ids[e][fi])]['labels']) as f:return f['material_xyz'],f['instance']
        for i in range(24):
            slot=int(perm[2,i]);a,b=banks[0][i],banks[2][slot]
            if v=='rich':edge_rows.append(dict(asset=i,rear=int(slot%3==0),source_centers=len(a['xyz']),target_centers=len(b['xyz']),near_material_edges=int((np.linalg.norm(a['q'][:,None]-b['q'][None],axis=-1)<=.015).sum())))
            T=poses[2,i]@inverse(poses[0,i]);p=apply(T,a['xyz'])
            for optimize in [False,True]:
                r=directed_match(p,a['rgb'],views[2][slot],slot,c,optimize);active=r['valid']&(r['kernel']>1e-6);correct=np.zeros(len(p),bool)
                for h,frame in enumerate(views[2][slot]):
                    ix=np.flatnonzero(active&(r['frame']==h))
                    if not len(ix):continue
                    qgrid,idgrid=label(2,findex[2][id(frame)]);uv=np.rint(r['uv'][ix]).astype(int);uv[:,0]=np.clip(uv[:,0],0,frame['depth'].shape[1]-1);uv[:,1]=np.clip(uv[:,1],0,frame['depth'].shape[0]-1);correct[ix]=(idgrid[uv[:,1],uv[:,0]]==a['physical_id'][ix])&(np.linalg.norm(qgrid[uv[:,1],uv[:,0]]-a['q'][ix],axis=1)<=.015)
                rows.append(dict(variant=v,asset=i,rear=int(slot%3==0),optimize_offset=int(optimize),source_centers=len(p),valid_source_queries=int(r['valid'].sum()),active_queries=int(active.sum()),correct_material_queries=int(correct.sum()),correct_fraction=float(correct.sum()/max(active.sum(),1)),offset_median_px=float(np.median(np.linalg.norm(r['offset'][active],axis=1))) if active.any() else None))
    write_csv(root/'true_pose_continuous_support.csv',rows);write_csv(root/'all512_material_edges.csv',edge_rows);rear=[r for r in edge_rows if r['rear']];assert sum(r['near_material_edges'] for r in rear)==33
    summary=[]
    for v in c['variants']:
        for optimize in [0,1]:
            rr=[r for r in rows if r['variant']==v and r['rear'] and r['optimize_offset']==optimize];summary.append(dict(variant=v,optimize_offset=optimize,rear_source_queries=sum(r['source_centers'] for r in rr),rear_valid_queries=sum(r['valid_source_queries'] for r in rr),rear_correct_material_queries=sum(r['correct_material_queries'] for r in rr),rear_with_three_correct_material_queries=sum(r['correct_material_queries']>=3 for r in rr)))
    save(root/'continuous_support_summary.json',dict(all512_rear_near_material_edges=33,source_centers_per_slot=512,summary=summary,scope='Truthblind inference sealed first. 33 raw-near pairs are preserved input geometry, NOT33 inferred correct matches. True-pose/true-ID query support is diagnostic, NOT task success or observability proof. Geometry edge audit runs once onrich, appearances correlated.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
