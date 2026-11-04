"""Postseal decomposition only; does not create or retune predictions."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save


def main():
    root=Path('results/shared_material_incidence_v1_run2');rows=read(root/'scores.json');graphs=read(root/'graph_diagnostics_postseal.json');lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows};result={}
    for family in ['catalogue_plane','material_shared']:
        rr=[r for r in rows if r['family']==family and r['method']=='original'];entry={}
        for geometry in ['aligned','offgrid']:
            entry[geometry]={m:sum(r['success'] for r in rows if r['family']==family and r['geometry']==geometry and r['method']==m) for m in ['original','sparse_unordered','graph_independent','graph_shared','graph_shared_no_domain']}
        gg=[g for g in graphs if g['family']==family and lookup[(g['scene'],g['asset'],family,'original')]['both_ids']];dist=[d for g in gg for d in g['face_nearest_generator_plane_distances_m']];entry['correct_ID_graph_plane_diagnostic']={'n_graphs':len(gg),'faces':len(dist),'faces_farther_than_6mm_from_ANY_generator_plane':sum(d>.006 for d in dist),'fraction_farther_than_6mm':float(np.mean(np.array(dist)>.006)),'median_distance_m':float(np.median(dist)),'not_true_surface_precision_hidden_generator_planes_included':True}
        for method in ['graph_independent','graph_shared','graph_shared_no_domain']:
            changes=[]
            for old in rr:
                new=lookup[(old['scene'],old['asset'],family,method)]
                if old['success']==new['success']:continue
                changes.append({'scene':old['scene'],'asset':old['asset'],'type':'gain' if new['success'] else 'loss','original_error_m':old['error_m'],'new_error_m':new['error_m'],'new_epsilon_m':new['epsilon_m'],'original_error_distance_to_threshold_m':abs(old['error_m']-.05),'new_error_distance_to_threshold_m':abs(new['error_m']-.05),'faces':new['faces'],'edges':new['edges']})
            entry[method+'_changed_vs_original']=changes
        result[family]=entry
    save(root/'mechanism_decomposition.json',{'families':result,'postseal_diagnostic_not_retuning':True,'generic_dictionary_span_prior_not_novelty_evidence':True});print(result)


if __name__=='__main__':main()
