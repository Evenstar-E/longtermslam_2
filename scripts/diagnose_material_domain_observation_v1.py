"""Post-seal identify whether coverage, spatial domain or sharing caused gains."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.material_domain_observation_v1 import cut_energy,shared_graph
from inspection_decisions.ray_conditioned_plane_polytope_v1 import intersection_bounds


def main():
    root=Path('results/material_domain_observation_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);planes=Path(cfg['plane_root'])
    for p,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/p)==h
    old=read(planes/'mechanism_diagnosis.json');selected=[];holes=[];scene_rows=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag;src=base/tag;pred=read(dest/'predictions.json');arrays=np.load(dest/'costs.npz')
        with (dest/'graphs.pkl').open('rb') as f:graphs=pickle.load(f)
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (planes/tag/'history_patches.pkl').open('rb') as f:patches=pickle.load(f)
        scene_rows.append({'scene':tag,'independent_same_coverage':pred['independent_domains']['selection']==pred['coverage_only']['selection'],
            'shared_same_independent':pred['shared_material_domain']['selection']==pred['independent_domains']['selection'],
            'shared_same_coverage':pred['shared_material_domain']['selection']==pred['coverage_only']['selection'],
            'shared_refinement_extra_integer':sum(v['extra_energy_integer'] for v in read(dest/'shared_pair_details.json')),
            'cross_edges_evaluated':sum(v['cross_edges'] for v in read(dest/'shared_pair_details.json'))})
        for method,result in pred.items():
            if not result['selection']:continue
            for i,k,l in result['selection']:
                c=float(arrays[f'coverage_{i}'][k,l]);d=float(arrays[f'independent_{i}'][k,l])
                selected.append({'scene':tag,'asset':i,'method':method,'coverage_cost':c,'independent_cost':d,'domain_minus_coverage':d-c,
                                 'shared_exact_if_computed':bool(arrays[f'shared_exact_{i}'][k,l])})
        bad=[v for v in old['bad_true_patches'] if v['scene']==tag]
        if not bad:continue
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        for item in bad:
            i=item['asset'];j=item['patch_index'];bb=inp['branches'][i]
            correct=[next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1]]
            p=patches[i][j]
            for witness in item['false_true_plane_hit_witnesses']:
                t=witness['epoch']-1;v=witness['view'];u,y=witness['target_pixel'];frame=np.load(src/f'frame_{t+1}_{v}.npz')
                M=truth['poses'][t,i]@np.linalg.inv(inp['T0'][i])@p['source_camera'];relative=np.linalg.inv(M)@frame['C'];o=relative[:3,3]
                rc=np.array([(u-(cfg['width']-1)/2)/cfg['focal_px'],(y-(cfg['height']-1)/2)/cfg['focal_px'],1.]);r=rc@relative[:3,:3].T
                bounds=intersection_bounds(p,o,r[None]);n=p['normals'].mean(0);z=(1-n@o)/(n@r);point=o+z*r
                uv=point[:2]/point[2]*p['focal']+p['principal']-p['center_uv'];g=graphs[i][t][correct[t]][j]
                distances=np.linalg.norm(g['x']-uv,axis=1);node=int(np.argmin(distances));assert distances[node]<1e-9
                energy,label=cut_energy(g,return_labels=True);a,b=graphs[i][0][correct[0]][j],graphs[i][1][correct[1]][j];joined=shared_graph(a,b,cfg);ej,zj=cut_energy(joined,return_labels=True)
                jj=int(np.argmin(np.linalg.norm(joined['x']-uv,axis=1)))
                inf=int(np.abs(g['unary']).sum()+sum(w for _,_,w in g['edges'])+1)
                g0={**g,'unary':g['unary'].copy()};g0['unary'][node]+=inf
                g1={**g,'unary':g['unary'].copy()};g1['unary'][node]-=inf
                absent,za=cut_energy(g0,return_labels=True);present,zp=cut_energy(g1,return_labels=True);present+=inf
                assert not za[node] and zp[node]
                holes.append({'scene':tag,'asset':i,'true_pair':correct,'source_pixel':p['pixel'],'target_pixel':[u,y],
                    'certain_free':bool(bounds['valid'][0] and bounds['upper'][0]<frame['depth'][y,u]-cfg['depth_tolerance_m']),
                    'independent_label_material_exists':bool(label[node]),'shared_label_material_exists':bool(zj[jj]),
                    'historical_positive_labels_all_one':bool(label[g['anchors']].all()),'independent_energy_integer':int(energy),'shared_energy_integer':int(ej),
                    'counterfactual_energy_force_absent':int(absent),'counterfactual_energy_force_present':int(present),
                    'finite_true_pair':bool(np.isfinite(arrays[f'independent_{i}'][tuple(correct)]))})
    panel=read(root/'previous_failure_panel.json');corrections={m:sum(v['new_results'][m]['identity_and_material_5cm'] for v in panel['quantification_15']) for m in ['ordinary_nn','coverage_only','independent_domains','shared_material_domain']}
    summary={'old_15_corrected':corrections,'shared_extra_integer_all_evaluated':sum(s['shared_refinement_extra_integer'] for s in scene_rows),
        'cross_edges_all_evaluated':sum(s['cross_edges_evaluated'] for s in scene_rows),
        'scenes_shared_same_independent':sum(s['shared_same_independent'] for s in scene_rows),'scenes_shared_same_coverage':sum(s['shared_same_coverage'] for s in scene_rows),
        'concavity_conditions_finite':sum(h['finite_true_pair'] for h in holes),
        'concavity_independent_domain_carved_hole':sum(not h['independent_label_material_exists'] for h in holes),
        'concavity_shared_domain_carved_hole':sum(not h['shared_label_material_exists'] for h in holes)}
    save(root/'mechanism_diagnosis.json',{'GT_assisted_concavity_checks':True,'updates_predictions':False,'summary':summary,'scene_mechanism':scene_rows,
        'selected_energy_decomposition':selected,'concavity_domain_labels':holes,'script_sha256':sha(Path(__file__))});print(summary,flush=True)


if __name__=='__main__':main()
