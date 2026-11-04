"""Index completed observation experiment, postseal diagnosis and rejected run."""
from pathlib import Path
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/set_valued_material_edges_v1');assert read(root/'audit.json')['passed']
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    reports=[Path('research/finite_edge_support_results_2026-10-25.md'),Path('research/finite_edge_support_independent_audit_2026-10-25.md'),Path('research/sparse_ray_material_incidence_next_v1_2026-10-25.md')]
    post=[root/'audit.json',root/'physical_geometry_gate.json',root/'oracle_edge_visibility.json',root/'oracle_edge_visibility_v2.json',root/'single_identity_negative_gate.json',Path('results/edge_conditioned_material_support_v1/aborted_status.json')]
    sources=[Path('scripts/diagnose_set_valued_material_edges_v1.py'),Path('scripts/diagnose_material_edge_visibility_oracle_v1.py'),Path('scripts/diagnose_material_edge_visibility_oracle_v2.py'),Path('scripts/audit_set_valued_material_edges_v1.py'),Path(__file__)]
    save(root/'final_evidence_index.json',{'complete_observation_experiment':True,'complete_new_end_to_end_192_task_accuracy_experiment':False,'old_task_metrics_unchanged':{'material_shared':80,'material_independent':68,'catalogue_plane':115},'all_jobs_finished':True,'prediction_seal_sha256':sha(root/'prediction_seal.json'),'reports_and_postseal_diagnostics':{str(p):sha(p) for p in reports+post+sources},'initial_single_direction_model_rejected':True,'literature_originality_confirmed':False,'verified_new_task_accuracy_gain':False,'claims':{'observed_segments':2629,'actual_one_axial_edge_segments':1607,'conditional_intervals_cover_actual_edge':1605,'unrestricted_support_escape_is_constructive_not_new_information':True,'minimum_K_is_shape_prior':True,'forced_same_is_shared_scalar_not_physical_ID':True,'oracle_not_inference':True}})
    print('INDEXED',flush=True)


if __name__=='__main__':main()
