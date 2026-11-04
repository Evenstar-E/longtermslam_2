"""Verify old evidence unchanged, seal both new experiment stages together."""
import json,csv
from pathlib import Path
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    checks=0
    for file in ['results/material_native_modes_comparison_v6/completion_integrity.json','results/material_observation_probe_v7/completion_integrity.json']:
        index=json.loads(Path(file).read_text(encoding='utf-8'))
        for group in ['source_hashes','evidence_hashes']:
            for p,h in index[group].items():assert sha(p)==h,p;checks+=1
    for folder,file in [('results/material_continuous_correspondence_comparison_v4','prediction_seal.json'),('results/material_joint_shared_comparison_v5','prediction_seal.json'),('results/material_native_correspondences_v6','correspondence_seal.json'),('results/material_native_modes_comparison_v6','prediction_seal.json'),('results/material_mode_intersection_v6','prediction_seal.json'),('results/material_observation_native_v7','correspondence_seal.json'),('results/material_observation_probe_v7','prediction_seal.json')]:
        folder=Path(folder);s=json.loads((folder/file).read_text(encoding='utf-8'))
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    for folder in [Path('data/textured_material_v1'),Path('data/textured_material_v2_train_probe')]:
        s=json.loads((folder/'release_manifest.json').read_text(encoding='utf-8'))
        for p,h in s.get('source_hashes',{}).items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    official=Path('third_party/material_local_matching_v1');s=json.loads((official/'source_manifest.json').read_text(encoding='utf-8'))
    for p,r in s['files'].items():assert sha(official/p)==r['sha256'];checks+=1
    roots=[Path('results/material_partial_track_transport_v7'),Path('results/material_partial_track_fixed_measure_v8')]
    for root in roots:
        seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
        for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
        for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
        rows=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')));assert len(rows)==864 and sum(r['method'] in ['ordinary_union','visibility_pair','history_union','shared_transport'] for r in rows)==384;checks+=2
    score192='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';assert sha('results/surface_measure_first_return_v1/scores.json')==score192;checks+=1
    report=Path('research/material_partial_track_transport_results_v8_2026-10-26.md');assert report.is_file();checks+=1
    files=[p for root in roots for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,Path('research/material_partial_track_transport_model_v7_2026-10-26.md'),Path('results/material_partial_track_transport_unit_v7.json'),Path('results/material_fixed_measure_check_v8.json'),Path('results/material_fixed_measure_counterexample_v8.json')]
    sources=['inspection_decisions/material_partial_track_transport_v7.py','inspection_decisions/material_partial_track_fixed_measure_v8.py','scripts/run_material_partial_track_transport_v7.py','scripts/run_material_partial_track_fixed_measure_v8.py','scripts/check_material_partial_track_transport_v7.py','scripts/check_material_fixed_measure_v8.py','scripts/score_material_partial_track_transport_v7.py','scripts/score_material_partial_track_fixed_measure_v8.py','scripts/diagnose_material_partial_track_transport_v7.py','scripts/prepare_material_fixed_measure_v8.py','scripts/report_material_partial_track_transport_v8.py','scripts/plot_material_partial_track_comparison_v8.py',__file__,'configs/material_partial_track_transport_v7.json','configs/material_partial_track_fixed_measure_v8.json']
    save(roots[-1]/'completion_integrity.json',dict(complete=True,checks=checks,old_evidence_unchanged=True,old_192_score_sha=score192,TRAIN_scenes=1,primary_rows_per_stage=384,no_new_training=True,independent_confirmation=False,figure_opened_and_inspected=True,figure_review='All four methods and appearances in both stages, zero counts labeled, identical 0-24 scales, N=1 TRAIN stated, actual candidate-pool comparability limit stated. No clipping observed.',source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files}));print('PARTIAL TRACK V8 COMPLETION',checks)

if __name__=='__main__':main()
