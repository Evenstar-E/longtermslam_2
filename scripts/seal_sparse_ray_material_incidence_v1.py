"""Final evidence index plus retention accounting; preserves inference seal."""
from pathlib import Path
import sys
import numpy as np
import scipy,open3d,shapely
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/sparse_ray_material_incidence_v1_run4');assert read(root/'audit.json')['passed']
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    rows=read(root/'scores.json');lookup={(r['scene'],r['asset'],r['family'],r['method']):r for r in rows};retention=[]
    for family in ['catalogue_plane','material_shared']:
        rr=[r for r in rows if r['family']==family and r['method']=='sparse_unordered']
        for control in ['original','plane','sparse_ordered']:
            gained=[r for r in rr if r['identity_and_material_5cm'] and not lookup[(r['scene'],r['asset'],family,control)]['identity_and_material_5cm']];lost=[r for r in rr if not r['identity_and_material_5cm'] and lookup[(r['scene'],r['asset'],family,control)]['identity_and_material_5cm']]
            retention.append({'family':family,'control':control,'gained':len(gained),'lost':len(lost),'net':len(gained)-len(lost),'gained_at_epsilon_zero':sum(abs(r['epsilon_m'])<1e-12 for r in gained),'gained_with_actual_displacement_update':sum(abs(r['epsilon_m'])>=1e-12 for r in gained)})
    save(root/'retention_decomposition.json',retention);save(root/'environment.json',{'Python':sys.version.split()[0],'NumPy':np.__version__,'SciPy':scipy.__version__,'Open3D':open3d.__version__,'Shapely':shapely.__version__,'GEOS':shapely.geos_version_string})
    reports=[Path('research/sparse_ray_material_incidence_results_2026-10-25.md'),Path('research/sparse_ray_material_incidence_independent_audit_2026-10-25.md'),Path('research/shared_material_incidence_next_v1_2026-10-25.md')]
    scripts=[Path('scripts/score_sparse_ray_material_incidence_v1.py'),Path('scripts/audit_sparse_ray_material_incidence_v1.py'),Path('scripts/diagnose_sparse_ray_material_geometry_v1.py'),Path('scripts/diagnose_sparse_ray_material_associations_v1.py'),Path(__file__)]
    attempts=[Path('results/sparse_ray_material_incidence_v1/failed_attempt.json'),Path('results/sparse_ray_material_incidence_v1_run2/interrupted_status.json'),Path('results/sparse_ray_material_incidence_v1_run3/interrupted_status.json')]
    post=[p for p in root.glob('*.json') if p.name not in read(root/'prediction_seal.json')['sha256'] and p.name!='final_evidence_index.json']
    save(root/'final_evidence_index.json',{'complete':True,'full_192_task_restricted_refinement_completed':True,'new_full_SE3_identity_joint_solver_completed':False,'inference_seal_sha256':sha(root/'prediction_seal.json'),'reports_postseal_scripts_attempts':{str(p):sha(p) for p in reports+scripts+attempts+post},'all_jobs_finished':True,'familiar_simulation_DEV_not_new_real_cross_period_validation':True,'overall_new_algorithm_gain_confirmed':False,'literature_originality_confirmed':False,'scope':{'identity_frozen':True,'rotation_frozen':True,'epoch1_frozen':True,'epoch2_single_direction_only':True},'results':{'catalogue_initial_original':115,'catalogue_initial_same_pool_plane':86,'catalogue_initial_sparse_ordered':82,'catalogue_initial_sparse_unordered':108,'catalogue_original_gained':6,'catalogue_original_lost':13,'shared_initial_original':80,'shared_initial_sparse_unordered':68},'coverage_correction':{'ordered_source_role_tasks':68,'unordered_role_tasks':192,'previous_observation_deficiency_claim_is_incomplete':True,'GT_diagnostic_not_estimator_input':True},'association_oracle_panel_is_not_algorithm_accuracy':True})
    print('INDEXED',flush=True)


if __name__=='__main__':main()
