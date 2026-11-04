"""Post-seal: actual dense retrieval, global generation and budget pruning."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_material_surface_distribution_v15b import configuration,load
from scripts.prepare_material_surface_fields_v15 import load_fields
from scripts.run_material_continuous_surface_correspondence_v14c import setup,dense_records
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from inspection_decisions.material_surface_distribution_v15b import dense_modes
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h,p
    ic,frames,ids,_,_,_,views,banks,_=setup(c,'rich');dense=dense_records(views);load_fields(c,'rich',banks,dense);scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);private_chart_material(scene,banks,ids);records=json.loads((scene/'frames.json').read_text(encoding='utf-8'));findex=[{id(f):k for k,f in enumerate(row)} for row in frames];raw=load(root/'rich_pair_proposals.npz');rows=[]
    @lru_cache(maxsize=16)
    def labels(e,k):
        with np.load(scene/records[int(ids[e][k])]['labels']) as z:return z['material_xyz'],z['instance']
    for i,q in enumerate(anchors):
        slot=int(perm[2,i]);a=banks[0][i];b=dense[2][slot];targetq=np.zeros_like(b['xyz']);targetid=np.full(len(targetq),-1,int)
        for fi,f in enumerate(views[2][slot]):
            h=b['frame']==fi;uv=b['uv'][h].astype(int);qq,ii=labels(2,findex[2][id(f)]);targetq[h]=qq[uv[:,1],uv[:,0]];targetid[h]=ii[uv[:,1],uv[:,0]]
        valid=np.flatnonzero(targetid==i);qt=cKDTree(targetq[valid]);support=qt.query_ball_point(a['q'],.015);m=dense_modes(a,b,c);native=cKDTree(b['xyz']);_,peakix=native.query(m['point']);correct=(targetid[peakix]==a['physical_id'][m['source']])&(np.linalg.norm(targetq[peakix]-a['q'][m['source']],axis=1)<=.015);peak_sources=np.bincount(m['source'],weights=correct,minlength=512)>0;region_sources=np.zeros(512,bool);gaps=[]
        for s,near in enumerate(support):
            if not len(near):continue
            jj=valid[np.asarray(near,int)];h=m['source']==s;distance=np.linalg.norm(b['xyz'][jj,None]-m['point'][h][None],axis=-1);region_sources[s]=(distance<=c['mode_radius_m']).any();besttrue=float(np.max(a['desc'][s]@b['desc'][jj].T));gaps.append(float(m['cosine'][h].max())-besttrue)
        selected=[]
        for s in np.flatnonzero(peak_sources):
            h=np.flatnonzero(correct&(m['source']==s));selected.append(int(h[m['cosine'][h].argmax()]))
        selected=np.array(selected,int);T=rigid(a['xyz'][m['source'][selected]],m['point'][selected],np.ones(len(selected))) if len(selected)>=3 else None;x=apply(poses[0,i],q);y=apply(poses[2,i],q)
        def err(T):return float(np.quantile(np.linalg.norm(apply(T,x)-y,axis=1),.95))
        row=dict(asset=i,rear=int(slot%3==0),source_centers=512,dense_material_supported_queries=sum(bool(h) for h in support),peak_supported_queries=int(peak_sources.sum()),region_supported_queries=int(region_sources.sum()),feature_best_vs_best_true_gap_median=float(np.median(gaps)) if gaps else None,oracle_peak_association_pair_error_m=err(T) if T is not None else None)
        for family in ['point','region']:
            for stage in ['raw','selected']:
                key=f'1_{i}_{slot}_{family}'+('_raw' if stage=='raw' else '');U=raw.get(key);E=np.array([err(t) for t in U]) if U is not None else np.array([]);row[f'{family}_{stage}_models']=len(E);row[f'{family}_{stage}_best_pair_error_m']=float(E.min()) if len(E) else None;row[f'{family}_{stage}_pair_qualified']=int(len(E)>0 and E.min()<=.05)
        rows.append(row)
    write_csv(root/'dense_global_correspondence_diagnosis.csv',rows);rear=[r for r in rows if r['rear']];summary=dict(rear_assets=8,dense_supported=sum(r['dense_material_supported_queries'] for r in rear),peak_supported=sum(r['peak_supported_queries'] for r in rear),region_supported=sum(r['region_supported_queries'] for r in rear),rear_peak_oracle_pair_qualified=sum(r['oracle_peak_association_pair_error_m'] is not None and r['oracle_peak_association_pair_error_m']<=.05 for r in rear))
    for family in ['point','region']:
        for stage in ['raw','selected']:summary[f'{family}_{stage}_pair_qualified']=sum(r[f'{family}_{stage}_pair_qualified'] for r in rear)
    save(root/'dense_global_diagnosis_summary.json',dict(summary=summary,scope='Correct H2 identity and private material coordinates AFTER prediction seal; dense native supported vs top3 peaks vs region union. Raw/selected are actual saved generated transforms; pair5cm is NOT full three-segment task success. GT-filtered peak rigid fit diagnostic never enters predictions; no observability proof or independent scene.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
