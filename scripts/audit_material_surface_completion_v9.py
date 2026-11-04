"""Final evidence/source preservation index, including unscored failed attempt."""
import json,re
from pathlib import Path
from scripts.run_textured_material_comparison_v1 import sha,save

def main():
    root=Path('results/material_surface_patch_registration_v9_run2');checks=0
    old=json.loads(Path('results/material_partial_track_fixed_measure_v8/completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['source_hashes','evidence_hashes']:
        for p,h in old[group].items():assert sha(p)==h,p;checks+=1
    for folder,file in [(root,'prediction_seal.json'),(Path('results/material_surface_geometry_v9'),'surface_seal.json')]:
        s=json.loads((folder/file).read_text(encoding='utf-8'));assert s['complete'] and not s['private_labels_read'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    for data in [Path('data/textured_material_v1'),Path('data/textured_material_v2_train_probe')]:
        s=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
        for p,h in s['files'].items():assert sha(data/p)==h;checks+=1
    failure=Path('results/material_surface_patch_registration_v9');assert not (failure/'prediction_seal.json').exists() and not list(failure.glob('*.npz'));s=json.loads((failure/'frozen_protocol.json').read_text(encoding='utf-8'));h=next(h for p,h in s['sources'].items() if p.endswith('run_material_surface_patch_registration_v9.py'));assert sha(failure/'runner_before_index_fix.py.txt')==h;checks+=2
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==480 and scores['total_rows']==672 and scores['max_selected_objective_error']<1e-8;checks+=1
    report=Path('research/material_surface_patch_results_v9_2026-10-26.md');model=Path('research/material_surface_patch_model_v9_2026-10-26.md')
    for file in [report,model]:
        for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if not link.startswith('https://'):assert (file.parent/link).is_file(),link;checks+=1
    score192='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';assert sha('results/surface_measure_first_return_v1/scores.json')==score192;checks+=1
    evidence=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+list(failure.iterdir())+[p for p in Path('results/material_surface_geometry_v9').iterdir() if p.is_file()]+[report,model,Path('results/material_surface_patch_check_v9.json')]
    source=['inspection_decisions/material_surface_patch_registration_v9.py','configs/material_surface_patch_registration_v9.json','scripts/check_material_surface_patch_registration_v9.py','scripts/cache_material_surface_geometry_v9.py','scripts/run_material_surface_patch_registration_v9.py','scripts/score_material_surface_patch_registration_v9.py','scripts/report_material_surface_patch_registration_v9.py','scripts/diagnose_material_surface_pose_v9.py',__file__]
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=480,TRAIN_scenes=1,new_training=False,independent_confirmation=False,old_evidence_unchanged=True,old_192_score_sha=score192,figure_png_opened_and_reviewed=True,figure_review='All5methods/4appearances, zeros labeled; actual success separate from known-ID candidate oracle with rear denominator; N=1 TRAIN and published-backend limitation clear; no clipping.',source_hashes={p:sha(p) for p in source},evidence_hashes={str(p):sha(p) for p in evidence}));print('SURFACE V9 INTEGRITY',checks)

if __name__=='__main__':main()
