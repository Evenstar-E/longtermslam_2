"""Post-seal GT material-support graph; never new inference candidates."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,setup
from scripts.score_material_surface_correspondence_v13c import private_chart_material
from scripts.score_textured_material_comparison_v1 import write_csv,material_anchors
from inspection_decisions.textured_material_matching_v1 import rigid,apply
from inspection_decisions.material_surface_patch_registration_v9 import inverse
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene)
    _,_,ids,_,_,_,_,banks,_=setup(c,'rich');private_chart_material(scene,banks,ids);rows=[]
    for i,q in enumerate(anchors):
        a,b,d=banks[0][i],banks[1][int(perm[1,i])],banks[2][int(perm[2,i])];trees=[cKDTree(x['q']) for x in [a,b,d]];ab,j=trees[1].query(a['q']);ad,k=trees[2].query(a['q']);bd,l=trees[2].query(b['q']);ba,_=trees[0].query(b['q']);fit=[]
        for x,y,dist,index in [(a,b,ab,j),(a,d,ad,k),(b,d,bd,l)]:
            active=dist<=.015;fit.append(rigid(x['xyz'][active],y['xyz'][index[active]],np.ones(active.sum())) if active.sum()>=3 else None)
        x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
        direct=physical_error(fit[0],fit[1],x,y1,y2) if fit[0] is not None and fit[1] is not None else None
        bridge=physical_error(fit[0],fit[2]@fit[0],x,y1,y2) if fit[0] is not None and fit[2] is not None else None
        rows.append(dict(asset=i,rear=int(perm[2,i]%3==0),h1_supported_source_nodes=int((ab<=.015).sum()),h2_supported_source_nodes=int((ad<=.015).sum()),historical_nodes_seen_in_both=int(((ab<=.015)&(ad<=.015)).sum()),c1_nodes_seen_c2=int((bd<=.015).sum()),c1_nodes_seen_c2_without_h_neighbor=int(((bd<=.015)&(ba>.015)).sum()),gt_association_direct_error_m=direct,gt_association_bridge_error_m=bridge))
    write_csv(root/'material_bridge_support.csv',rows);rr=[r for r in rows if r['rear']]
    save(root/'material_bridge_support_summary.json',dict(rear_assets=len(rr),historical_triple_nodes=sum(r['historical_nodes_seen_in_both'] for r in rr),c1_c2_nodes_without_h_neighbor=sum(r['c1_nodes_seen_c2_without_h_neighbor'] for r in rr),direct_gt_association_qualified=sum(r['gt_association_direct_error_m'] is not None and r['gt_association_direct_error_m']<=.05 for r in rr),bridge_gt_association_qualified=sum(r['gt_association_bridge_error_m'] is not None and r['gt_association_bridge_error_m']<=.05 for r in rr),scope='True identity/material coordinates select pair associations after prediction seal. Source-bank samples within15mm, not exact same point and not unseen-native support proof. GT fits NEVER inserted into inference or counted as task success. Geometry audited once onrich; no new independent scenes.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
