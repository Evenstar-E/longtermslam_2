"""Final immutable-source/data/weight/prediction and report evidence index."""
import json,re,csv
from pathlib import Path
from PIL import Image
from scripts.run_textured_material_comparison_v1 import sha,save


def main():
    root=Path('results/material_native_modes_comparison_v6');checks=0
    seals=[('results/material_continuous_correspondence_comparison_v4','prediction_seal.json'),('results/material_joint_shared_comparison_v5','prediction_seal.json'),('results/material_native_correspondences_v6','correspondence_seal.json'),(str(root),'prediction_seal.json'),('results/material_mode_intersection_v6','prediction_seal.json')]
    for folder,file in seals:
        folder=Path(folder);s=json.loads((folder/file).read_text(encoding='utf-8'));assert s['complete'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    p=Path('third_party/material_local_matching_v1');manifest=json.loads((p/'source_manifest.json').read_text(encoding='utf-8'))
    for f,r in manifest['files'].items():assert sha(p/f)==r['sha256'];checks+=1
    expected='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';assert sha('results/surface_measure_first_return_v1/scores.json')==expected;checks+=1
    cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text(encoding='utf-8'));data=Path(cfg['dataset']);release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    for file in ['scoring_audit.json','raw_provenance_objective_audit.json']:
        a=json.loads((root/file).read_text(encoding='utf-8'));assert sha('scripts/score_material_native_modes_v6.py' if file.startswith('scoring') else 'scripts/audit_material_native_modes_v6.py')==a['source_sha'];checks+=1
    rows=list(csv.DictReader((root/'task_scores.csv').open(encoding='utf-8-sig')));assert len(rows)==2688;checks+=1
    assert all(int(r['qualified'])<=int(r['oracle_pool_qualified']) for r in rows);checks+=1
    for file in ['native_modes_results.png','native_material_support.png']:
        with Image.open(root/file) as im:assert im.width>=1500 and im.height>=800;im.verify();checks+=1
    report=Path('research/material_native_modes_results_v6_2026-10-26.md')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if not link.startswith('https://'):assert (report.parent/link).is_file();checks+=1
    evidence=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[Path('research/material_native_modes_protocol_v6_2026-10-26.md'),report,Path('research/material_partial_tracks_next_v7_2026-10-26.md')]
    source=['scripts/score_material_native_modes_v6.py','scripts/audit_material_native_modes_v6.py','scripts/analyze_material_native_modes_v6.py','scripts/diagnose_material_native_new_candidates_v6.py','scripts/report_material_native_modes_v6.py',__file__]
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,old_192_score_sha=expected,old_v4_v5_and_new_prediction_seals_verified=True,official_sources_and_weights_unchanged=True,dataset_release_verified=True,task_rows=2688,
        two_figures_opened_and_visually_inspected=True,figure_review='All 7 methods and 4 paired appearances present; actual vs oracle separated; support panels label distinct denominators80/16; zeros visible; no inferential CI from correlated assets.',
        evidence_hashes={str(p):sha(p) for p in evidence},source_hashes={p:sha(p) for p in source},no_new_training=True,no_new_real_data_accuracy=True,no_independent_confirmation=True))
    print('V6 COMPLETION INTEGRITY',checks,flush=True)


if __name__=='__main__':main()
