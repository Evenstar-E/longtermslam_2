"""Verify actual V14c seals, inherited unique evidence, and all task outputs."""
import json
from pathlib import Path
from scripts.run_material_continuous_surface_correspondence_v14c import configuration
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);checks=0;parent=Path(c['previous_root']);previous=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in previous[group].items():assert sha(p)==h,p;checks+=1
    protocol=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    for p,h in protocol['geometric_reuse']['blocks'].items():assert sha(p)==h,p;checks+=1
    for oldroot in ['results/material_continuous_surface_correspondence_v14','results/material_continuous_surface_correspondence_v14_run2']:
        frozen=json.loads((Path(oldroot)/'frozen_protocol.json').read_text(encoding='utf-8'))
        for p,h in frozen['sources'].items():assert sha(p)==h,p;checks+=1
        assert not (Path(oldroot)/'prediction_seal.json').exists() and not (Path(oldroot)/'scores.json').exists();checks+=1
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==768 and scores['source_centers_verified']==147456 and scores['max_selected_objective_error']<1e-8;checks+=1
    support=json.loads((root/'continuous_support_summary.json').read_text(encoding='utf-8'));assert support['all512_rear_near_material_edges']==33;checks+=1
    old192='results/surface_measure_first_return_v1/scores.json';assert sha(old192)=='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';checks+=1
    report=Path('research/material_continuous_surface_results_v14_2026-10-27.md');assert report.is_file() and (root/'task_comparison.png').is_file();checks+=1
    sources=list(seal['sources'])+['scripts/diagnose_material_continuous_support_v14.py','scripts/diagnose_material_continuous_triples_v14.py','scripts/diagnose_material_bridge_support_v14.py','scripts/diagnose_material_radiometry_v14.py','scripts/probe_material_continuous_solver_v14.py','scripts/score_material_continuous_solver_probe_v14.py','scripts/analyze_material_continuous_mainline_v14.py','scripts/report_material_continuous_surface_v14.py','scripts/check_material_track_information_v14.py','scripts/check_material_exposure_derivatives_v15.py','inspection_decisions/material_exposure_derivatives_v15.py',__file__]
    files=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,Path('research/material_continuous_surface_model_v14_2026-10-27.md'),Path('research/material_continuous_surface_decision_v14_2026-10-27.md'),Path('results/material_continuous_exposure_checks_v14c.json'),Path('results/material_track_information_v14.json'),Path('results/material_exposure_derivative_checks_v15.json')]
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=768,old_192_sha=sha(old192),old_evidence_unchanged=True,new_training=False,TRAIN_scenes=1,source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files},scope='Prediction and source integrity, not novelty/independent generalization confirmation. PNG visual review performed separately. V15 derivative is prototype only, not integrated task result.'))
    print('V14c COMPLETION',checks)


if __name__=='__main__':main()
