"""Audit provenance, exact candidate identity, relaxation and selected scores."""
from pathlib import Path
import pickle
import numpy as np
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.shared_material_incidence_v1 import score_graph,pixel_groups
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/shared_material_incidence_v1_run2');cfg=read(root/'config.json');old=Path(cfg['input_root']);seal=read(root/'prediction_seal.json');assert seal['complete'];checks=0;recomputed=0;maximum=0.;multiple=0;total_pixels=0;total_rays=0;selected_differences=0
    for name,h in seal['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h;checks+=1
        for name,h in read(old/'manifest.json')[field].items():assert sha(name)==h;checks+=1
    graphs_stats=[];scores=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];prior=read(old/tag/'predictions.json');current=read(root/tag/'predictions.json');graphs=read(root/tag/'graphs.json');assert len(prior)==len(current)==len(graphs)==48
        with (old/tag/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        for a,b,G in zip(current,prior,graphs):
            assert (a['asset'],a['family'])==(b['asset'],b['family'])==(G['asset'],G['family']);assert len(a['scan'])==len(b['scan']);assert a['selected_slots']==b['selected_slots'];checks+=3
            for method,pred in b['predictions'].items():assert a['predictions'][method]==pred;checks+=1
            i=a['asset'];slot=a['selected_slots'][1];rows=raw[2][slot];ids,n=pixel_groups(rows);counts=np.bincount(ids,minlength=n);multiple+=int((counts>1).sum());total_pixels+=n;total_rays+=len(rows)
            for x,y in zip(a['scan'],b['scan']):
                assert x['epsilon_m']==y['epsilon_m'];checks+=1
                for key in ['nn','plane','sparse_ordered','sparse_unordered']:assert x[key]==y[key];checks+=1
                assert x['graph_shared']>=x['graph_independent']-1e-12;assert x['graph_shared']>=x['graph_shared_no_domain']-1e-12;checks+=2
            for method in ['graph_independent','graph_shared','graph_shared_no_domain']:
                pred=a['predictions'][method];T2=np.array(a['initial_T2']);T2[:3,3]+=pred['epsilon_m']*T2[:3,2];assert np.allclose(T2,pred['T2'],atol=1e-14,rtol=0);minimum=min(s[method] for s in a['scan']);eligible=[j for j,s in enumerate(a['scan']) if s[method]<=minimum+cfg['selection_tolerance']];j=min(eligible,key=lambda j:abs(a['scan'][j]['epsilon_m']));assert pred['scan_index']==j;checks+=2
            selected_differences+=int(a['predictions']['graph_independent']['scan_index']!=a['predictions']['graph_shared']['scan_index'])
            # Recompute every selected shared score from raw observations, not JSON.
            pred=a['predictions']['graph_shared'];x=pose_vector(np.array(pred['T2']),inp['T0'][slot,:3,3]);fresh=score_graph(x,rows,G['graph'],cfg);row=a['scan'][pred['scan_index']]
            for name in ['independent','shared','shared_no_domain']:
                err=abs(fresh[name]-row['graph_'+name]);maximum=max(maximum,err);assert err<1e-11;checks+=1
            recomputed+=1;graphs_stats.append({'family':a['family'],**a['graph_counts']});scores.extend(s['graph_diagnostics']['contradictory_pixels'] for s in a['scan'])
    summary={'checks':checks,'selected_score_recomputations':recomputed,'max_recomputed_score_error':maximum,'all_old_candidates_scores_predictions_unchanged':True,'all_graph_relaxation_inequalities_passed':True,'source_pixels_across_two_families':total_pixels,'late_ray_pairs_across_two_families':total_rays,'multi_neighbor_pixels_across_two_families':multiple,'multi_neighbor_pixel_fraction':multiple/total_pixels,'shared_vs_independent_different_selections':selected_differences,'candidate_mean_contradictory_pixel_count':float(np.mean(scores)),'median_faces':float(np.median([g['faces'] for g in graphs_stats])),'median_edges':float(np.median([g['edges'] for g in graphs_stats])),'GT_read':False}
    save(root/'audit.json',summary);print(summary)


if __name__=='__main__':main()
