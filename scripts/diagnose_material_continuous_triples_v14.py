"""Actual dense-view shared support, not an inference or sparse-bank proxy."""
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
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));_,frames,ids,_,_,_,views,banks,_=setup(c,'rich');private_chart_material(scene,banks,ids);findex=[{id(f):h for h,f in enumerate(row)} for row in frames];rows=[]
    @lru_cache(maxsize=16)
    def label(e,fi):
        with np.load(scene/records[int(ids[e][fi])]['labels']) as f:return f['material_xyz'],f['instance']
    for i in range(24):
        a=banks[0][i]
        for optimize in [False,True]:
            correct=[];active=[]
            for e in [1,2]:
                slot=int(perm[e,i]);T=poses[e,i]@inverse(poses[0,i]);r=directed_match(apply(T,a['xyz']),a['rgb'],views[e][slot],slot,c,optimize);mask=r['valid']&(r['kernel']>1e-6);good=np.zeros(len(mask),bool)
                for h,f in enumerate(views[e][slot]):
                    ix=np.flatnonzero(mask&(r['frame']==h))
                    if not len(ix):continue
                    qgrid,idgrid=label(e,findex[e][id(f)]);uv=np.rint(r['uv'][ix]).astype(int);uv[:,0]=np.clip(uv[:,0],0,f['depth'].shape[1]-1);uv[:,1]=np.clip(uv[:,1],0,f['depth'].shape[0]-1);good[ix]=(idgrid[uv[:,1],uv[:,0]]==i)&(np.linalg.norm(qgrid[uv[:,1],uv[:,0]]-a['q'][ix],axis=1)<=.015)
                correct.append(good);active.append(mask)
            rows.append(dict(asset=i,rear=int(perm[2,i]%3==0),optimize_offset=int(optimize),h1_correct=int(correct[0].sum()),h2_correct=int(correct[1].sum()),both_active=int((active[0]&active[1]).sum()),both_correct=int((correct[0]&correct[1]).sum())))
    write_csv(root/'true_pose_continuous_triples.csv',rows);summary=[]
    for optimize in [0,1]:
        rr=[r for r in rows if r['rear'] and r['optimize_offset']==optimize];summary.append(dict(optimize_offset=optimize,rear_historical_source_nodes=4096,rear_both_active=sum(r['both_active'] for r in rr),rear_both_correct=sum(r['both_correct'] for r in rr),rear_assets_with_three_correct=sum(r['both_correct']>=3 for r in rr)))
    save(root/'continuous_triple_support_summary.json',dict(variant='rich',summary=summary,scope='True poses/IDs, actual dense native target views,15mm private material checks after prediction seal. No source-bank-only overlap proxy. Diagnostic, not algorithm success or proof of observability.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
