"""Old evidence preservation and new TRAIN-probe completion index."""
import csv,json,re
from pathlib import Path
from PIL import Image
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=json.loads(Path('configs/material_observation_probe_v7.json').read_text(encoding='utf-8'));root=Path(cfg['output']);data=Path(cfg['dataset']);checks=0
    # V6's complete index includes V4/V5-derived evidence and old next contracts.
    old=json.loads(Path('results/material_native_modes_comparison_v6/completion_integrity.json').read_text(encoding='utf-8'))
    for group in ['evidence_hashes','source_hashes']:
        for p,h in old[group].items():assert sha(p)==h;checks+=1
    for folder,file in [('results/material_continuous_correspondence_comparison_v4','prediction_seal.json'),('results/material_joint_shared_comparison_v5','prediction_seal.json'),('results/material_native_correspondences_v6','correspondence_seal.json'),('results/material_native_modes_comparison_v6','prediction_seal.json'),('results/material_mode_intersection_v6','prediction_seal.json'),(cfg['native_root'],'correspondence_seal.json'),(str(root),'prediction_seal.json')]:
        folder=Path(folder);s=json.loads((folder/file).read_text(encoding='utf-8'));assert s['complete'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    for folder in [Path('data/textured_material_v1'),data]:
        s=json.loads((folder/'release_manifest.json').read_text(encoding='utf-8'))
        for p,h in s.get('source_hashes',{}).items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    official=Path('third_party/material_local_matching_v1');manifest=json.loads((official/'source_manifest.json').read_text(encoding='utf-8'))
    for p,r in manifest['files'].items():assert sha(official/p)==r['sha256'];checks+=1
    score192='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';assert sha('results/surface_measure_first_return_v1/scores.json')==score192;checks+=1
    oldindex=json.loads(Path('data/textured_material_v1/dataset_index.json').read_text(encoding='utf-8'));newindex=json.loads((data/'dataset_index.json').read_text(encoding='utf-8'));assert len(newindex)==1 and newindex[0]['split']=='train';checks+=1
    for field in ['shape_keys','texture_keys']:assert not {k for r in oldindex for k in r[field]}&set(newindex[0][field]);checks+=1
    rows=list(csv.DictReader((root/'task_scores.csv').open(encoding='utf-8-sig')));coverage=list(csv.DictReader((root/'coverage.csv').open(encoding='utf-8-sig')));assert len(rows)==768 and len(coverage)==192;checks+=2
    for r in coverage:
        assert int(r['native_history_queries'])>=int(r['history_queries_physically_visible_both'])>=int(r['history_queries_sampled_both'])>=int(r['history_queries_matched_both']);checks+=1
    for file,source in [('scoring_audit.json','scripts/score_material_observation_probe_v7.py'),('raw_audit.json','scripts/audit_material_observation_probe_v7.py')]:
        a=json.loads((root/file).read_text(encoding='utf-8'));assert a['complete'] and sha(source)==a['source_sha'];checks+=1
    report=Path('research/material_observation_probe_results_v7_2026-10-26.md');contract=Path('research/material_visibility_partial_tracks_contract_v7b_2026-10-26.md')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if not link.startswith('https://'):assert (report.parent/link).is_file();checks+=1
    for file in ['acquisition_task_comparison.png','coverage_and_view_failures.png']:
        with Image.open(root/file) as im:assert im.width>=1500 and im.height>=800;im.verify();checks+=1
    evidence=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,contract,Path('results/material_partial_pair_registration_check_v7.json')]
    sources=[__file__,'scripts/build_material_observation_coverage_v7.py','scripts/cache_material_observation_native_v7.py','scripts/run_material_observation_registration_v7.py','scripts/score_material_observation_probe_v7.py','scripts/audit_material_observation_probe_v7.py','scripts/report_material_observation_probe_v7.py','scripts/check_material_partial_pair_registration_v7.py','inspection_decisions/material_observation_inputs_v7.py','inspection_decisions/material_partial_pair_registration_v7.py','configs/textured_material_dataset_v2.json','configs/material_observation_probe_v7.json','configs/material_observation_registration_v7.json']
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,task_rows=768,coverage_rows=192,TRAIN_scenes=1,new_model_trained=False,new_persistent_material_model_implemented=False,old_v1_v4_v5_v6_evidence_unchanged=True,official_sources_weights_unchanged=True,old_192_score_sha=score192,
        source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in evidence},two_figures_opened_and_visually_inspected=True,figure_review='All4 methods/4appearances/2capture conditions retained, zeros shown; N=1 TRAIN explicit; unequal capture budgets and front16/rear8 denominators explicit; no inferential CIs from correlated pixels/assets.'))
    print('V7 FINAL INTEGRITY',checks,flush=True)


if __name__=='__main__':main()
