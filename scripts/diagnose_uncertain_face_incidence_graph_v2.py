"""Postseal geometry support and improvement decomposition, no new inference."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/uncertain_face_incidence_graph_v2');cfg=read(root/'config.json');assert read(root/'prediction_seal.json')['complete'];parent=Path(cfg['parent_root']);scores=read(root/'scores.json');lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in scores};oldroot=Path('results/shared_material_incidence_v1_run2');events_path=Path(cfg['input_root'])/'physical_geometry_gate.json';events=read(events_path)['rows'];eventmap={}
    for e in events:
        if 2 not in e['key']:continue
        token=tuple(e['key'])+tuple(np.round(e['support_coordinates_m'],9));eventmap.setdefault((e['scene'],e['asset']),{}).setdefault(token,[]).append(e)
    geometry=read(root/'graph_geometry_postseal.json');plane_summary={};coverage=[];optimization={};changes=[]
    for family in ['catalogue_plane','material_shared']:
        plane_summary[family]={}
        for method in ['initial_graph']+cfg['methods']:
            gg=[g for g in geometry if g['family']==family and g['method']==method and g['both_ids']];dist=[v for g in gg for v in g['face_distances_to_any_generator_plane_m']];plane_summary[family][method]={'graphs':len(gg),'faces':len(dist),'farther_than_6mm':sum(d>.006 for d in dist),'fraction_farther_than_6mm':float(np.mean(np.array(dist)>.006)) if dist else None}
    def covered(graph,offsets,token):
        a,k=token[:2];da,dk=token[2:];axes=[f['axis'] for f in graph['faces']]
        for e in graph['edges']:
            f,g=e['faces']
            if (axes[f],axes[g])==(a,k) and np.hypot(offsets[f]-da,offsets[g]-dk)<=.006:return True
            if (axes[g],axes[f])==(a,k) and np.hypot(offsets[g]-da,offsets[f]-dk)<=.006:return True
        return False
    for run in read(root/'runs.json'):
        tag=run['scene'];oldgraphs={(g['asset'],g['family']):g['graph'] for g in read(oldroot/tag/'graphs.json')};prior=read(parent/tag/'predictions.json');new=read(root/tag/'predictions.json')
        for old,row in zip(prior,new):
            i=row['asset'];family=row['family'];original=lookup[(tag,i,family,'original')];graph=row['graph'];gg=oldgraphs[(i,family)];tracks=eventmap.get((tag,i),{});shared=[t for t,ee in tracks.items() if any(e['epoch']<2 for e in ee) and any(e['epoch']==2 for e in ee)]
            record={'scene':tag,'asset':i,'family':family,'both_ids':original['both_ids'],'true_shared_depth_supports':len(shared)}
            variants={'old_coordinate_graph':(gg,[f['offset'] for f in gg['faces']]),'normal_initial_graph':(graph,[f['offset'] for f in graph['faces']]),'normal_joint_graph':(graph,row['parameters']['normal_joint'][12:12+len(graph['faces'])]),'bounded_joint_graph':(graph,row['parameters']['bounded_joint'][12:12+len(graph['faces'])])}
            for name,(g,offsets) in variants.items():record[name+'_covered_supports']=sum(covered(g,offsets,t) for t in shared)
            coverage.append(record);optimization.setdefault(family,{'n':0,'v1_bounded_energy_worse_than_point':0,'v2_bounded_energy_worse_than_point':0});op=optimization[family];op['n']+=1;op['v1_bounded_energy_worse_than_point']+=old['logs']['bounded_joint']['final_cost']>old['logs']['normal_joint']['final_cost']+1e-10;op['v2_bounded_energy_worse_than_point']+=row['common_pool_selection']['bounded_joint']['energy']>row['common_pool_selection']['normal_joint']['energy']+1e-10
            for method in cfg['methods']:
                now=lookup[(tag,i,family,method)]
                if original['success']!=now['success']:changes.append({'scene':tag,'asset':i,'family':family,'method':method,'change':'gain' if now['success'] else 'loss','original_error_m':original['error_m'],'new_error_m':now['error_m'],'selected_endpoint':row['common_pool_selection'][method]['source']})
    coverage_summary={}
    for family in ['catalogue_plane','material_shared']:
        rr=[r for r in coverage if r['family']==family and r['both_ids']];coverage_summary[family]={'correct_ID_tasks':len(rr)}
        for name in ['old_coordinate_graph','normal_initial_graph','normal_joint_graph','bounded_joint_graph']:coverage_summary[family][name+'_tasks_with_any_true_shared_depth_support']=sum(r[name+'_covered_supports']>0 for r in rr)
    save(root/'mechanism_diagnostics.json',{'plane_summary':plane_summary,'support_coverage_summary':coverage_summary,'support_coverage_rows':coverage,'optimization_feasible_dominance':optimization,'changed_outcomes':changes,'true_support_event_file_sha256':sha(events_path),'GT_only_after_both_full_seals':True,'support_coverage_ignores_domain_spans_is_optimistic':True,'correlation_not_per_failure_causal_identification':True});print({'plane_summary':plane_summary,'support_coverage_summary':coverage_summary,'optimization':optimization})


if __name__=='__main__':main()
