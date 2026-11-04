"""Final independent indexes for both unchanged inference stages and reports."""
from pathlib import Path
import platform,numpy,scipy
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    roots=[Path('results/uncertain_face_incidence_graph_v1_run2'),Path('results/uncertain_face_incidence_graph_v2')];sources=[Path(__file__),Path('scripts/score_uncertain_face_incidence_graph_v1.py'),Path('scripts/audit_uncertain_face_incidence_graph_v1.py'),Path('scripts/audit_uncertain_face_incidence_graph_v2.py'),Path('scripts/diagnose_uncertain_face_incidence_graph_v2.py'),Path('scripts/diagnose_uncertain_face_endpoint_capacity_v2.py'),Path('scripts/check_uncertain_face_unseen_termination_v1.py'),Path('research/uncertain_face_unseen_termination_counterexample_v1.json'),Path('research/uncertain_face_incidence_results_2026-10-25.md'),Path('research/typed_material_incidence_next_v1_2026-10-25.md')]
    for root in roots:
        assert read(root/'prediction_seal.json')['complete'];assert (root/'scores.json').exists() and (root/'audit.json').exists();save(root/'environment.json',{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__});files=[p for p in root.rglob('*') if p.is_file() and p.name!='final_evidence_index.json'];save(root/'final_evidence_index.json',{'complete':True,'familiar_DEV_not_independent_confirmation':True,'physical_IDs_and_graph_topology_not_updated':True,'files':{str(p.relative_to(root)):sha(p) for p in files},'additional_sources_and_reports':{str(p):sha(p) for p in sources},'novelty_confirmed':False});print('INDEXED',root,len(files))


if __name__=='__main__':main()
