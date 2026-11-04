"""Final evidence index, inference seal and evaluator evidence kept distinct."""
from pathlib import Path
import platform
import numpy,scipy,shapely
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/shared_material_incidence_v1_run2');assert read(root/'prediction_seal.json')['complete'];audit=read(root/'audit.json');assert audit['selected_score_recomputations']==384
    assert all(x['n']==192 for f in read(root/'summary.json').values() for x in f.values())
    save(root/'environment.json',{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__})
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='final_evidence_index.json'];extra=[Path('scripts/score_shared_material_incidence_v1.py'),Path('scripts/audit_shared_material_incidence_v1.py'),Path('scripts/analyze_shared_material_incidence_v1.py'),Path(__file__),Path('research/shared_material_incidence_results_2026-10-25.md'),Path('research/uncertain_face_incidence_joint_next_v1_2026-10-25.md')]
    save(root/'final_evidence_index.json',{'complete_inference_and_evaluation':True,'familiar_DEV_not_independent_confirmation':True,'source_files':{str(p):sha(p) for p in extra},'files':{str(p.relative_to(root)):sha(p) for p in files},'initial_engineering_attempt_unsealed_unscored':'results/shared_material_incidence_v1','novelty_confirmed':False});print('FINAL INDEX',len(files))


if __name__=='__main__':main()
