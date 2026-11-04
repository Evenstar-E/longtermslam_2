"""Same measured512 centers: separate plane filtering and64-node quantization."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_material_surface_correspondence_v13b import configuration,setup
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from scripts.cache_material_surface_geometry_v9 import unpack
from inspection_decisions.material_surface_correspondence_v13 import chart_basis,farthest_indices
from inspection_decisions.textured_material_matching_v1 import apply,rigid
from scripts.score_material_native_modes_v6 import physical_error
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene)
    _,frames,ids,_,_,_,_,banks,_=setup(cfg,'rich');charts=unpack(Path(cfg['chart_cache'])/'rich_charts.npz');private_chart_material(scene,banks,ids);eligible={};fps={};rows=[];fits=[]
    for e,row in enumerate(banks):
        for i,a in enumerate(row):
            tree=cKDTree(a['xyz']);good=[]
            for h,x in enumerate(a['xyz']):
                frame=frames[e][int(a['frame'][h])];b=chart_basis(x,a['xyz'][tree.query_ball_point(x,cfg['chart_neighbor_radius_m'])],frame['C'],cfg)
                if b is not None:good.append(h)
            eligible[e,i]=np.array(good if good else list(range(len(a['xyz']))),int);fps[e,i]=farthest_indices(a['xyz'],cfg['chart_nodes'])
    for i,q in enumerate(anchors):
        j,k=map(int,perm[1:,i]);sets={}
        for family in ['all512','plane_eligible','plane_FPS64','unfiltered_FPS64']:
            sets[family]=[np.arange(len(banks[e][s]['xyz'])) if family=='all512' else eligible[e,s] if family=='plane_eligible' else charts[e][s]['center_index'] if family=='plane_FPS64' else fps[e,s] for e,s in enumerate([i,j,k])]
        for family,index in sets.items():
            TT=[]
            for e,target in [(1,j),(2,k)]:
                a,b=banks[0][i],banks[e][target];aa,bb=index[0],index[e];distance=np.linalg.norm(a['q'][aa,None]-b['q'][None,bb],axis=-1);near=distance<=.015;dist,match=cKDTree(b['q'][bb]).query(a['q'][aa]);rev=cKDTree(a['q'][aa]).query(b['q'][bb])[1];ok=(dist<=.015)&(rev[match]==np.arange(len(aa)));h=np.flatnonzero(ok);T=rigid(a['xyz'][aa[h]],b['xyz'][bb[match[h]]]);TT.append(T)
                rows.append(dict(asset=i,rear=int(perm[2,i]%3==0),pair=e-1,family=family,source_nodes=len(aa),target_nodes=len(bb),near_material_edges=int(near.sum()),mutual_material_pairs=len(h),rigid_fit_exists=int(T is not None)))
            x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);error=float(physical_error(*TT,x,y1,y2)) if all(T is not None for T in TT) else None;fits.append(dict(asset=i,rear=int(perm[2,i]%3==0),family=family,q_assisted_error_m=error,q_assisted_qualified=int(error is not None and error<=.05)))
    write_csv(root/'sampling_loss_nodes.csv',rows);write_csv(root/'sampling_loss_association_fits.csv',fits);summary=[]
    for family in sets:
        rr=[r for r in rows if r['family']==family and r['rear'] and r['pair']==1];ff=[r for r in fits if r['family']==family and r['rear']]
        summary.append(dict(family=family,rear_H2_near_edges=sum(r['near_material_edges'] for r in rr),rear_with_any_H2_near_edge=sum(r['near_material_edges']>0 for r in rr),rear_q_assisted_qualified=sum(r['q_assisted_qualified'] for r in ff),rear_mutual_pairs=sum(r['mutual_material_pairs'] for r in rr)))
    save(root/'sampling_loss_summary.json',dict(summary=summary,scope='Geometry once onrich; all4appearances share geometry, not4 independent replications. Labels read only AFTER seal. SAME original512 measured centers, plane eligibility vs actual64FPS vs unfiltered64FPS, all15mm qnear/GT-association fits are diagnostics NOT inference success. No source/prediction edits, no raw-data impossibility claim.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
