"""Preserve old evidence and independently seal the completed V12 artifacts."""
import json
from pathlib import Path
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);checks=0;parent=Path(cfg['retained_root'])
    old=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for g in ['source_hashes','evidence_hashes']:
        for p,h in old[g].items():assert sha(p)==h,p;checks+=1
    protocol=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));assert sha(parent/'completion_integrity.json')==protocol['parent_completion_sha'];checks+=1
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==768 and scores['max_selected_objective_error']<1e-8;checks+=1
    for v in cfg['variants']:
        d=json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8'));assert d['complete'] and not d['private_labels_read'];checks+=1
    old192='results/surface_measure_first_return_v1/scores.json';assert sha(old192)=='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';checks+=1
    report=Path('research/material_inverse_plane_pose_results_v12_2026-10-27.md');assert report.is_file() and (root/'task_comparison.png').is_file();checks+=1
    sources=list(seal['sources'])+['scripts/diagnose_material_full_propagation_v12.py','scripts/diagnose_material_local_information_v12.py','scripts/analyze_material_mainline_decision_v12.py','scripts/report_material_inverse_plane_pose_proposals_v12.py','scripts/check_material_inverse_plane_pose_math_v12.py',__file__]
    files=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,Path('research/material_inverse_plane_pose_model_v12_2026-10-27.md'),Path('results/material_inverse_plane_pose_checks_v12.json'),Path('results/material_inverse_plane_pose_math_check_v12.json')]
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=768,old_evidence_unchanged=True,old_192_sha=sha(old192),new_pose_generation=True,new_training=False,TRAIN_scenes=1,source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files},figure_review='PNG opened separately; same0-24axes, all methods/appearances/counts, no fake CI; PDF exported.'))
    print('V12 COMPLETION',checks)


if __name__=='__main__':main()
