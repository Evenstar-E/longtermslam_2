"""Read-only verification of scientific evidence; only writes a new V4 receipt."""
import json
from pathlib import Path
from scripts.run_textured_material_comparison_v1 import sha,save


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    v4=load('configs/material_continuous_correspondence_v4.json');cfg=load(v4['base_config']);root=Path(v4['comparison_root']);checks=0;anchors={}
    def check(path,expected):
        nonlocal checks
        assert sha(path)==expected,str(path);checks+=1
    def verify(folder,name):
        folder=Path(folder);s=load(folder/name);assert s['complete'];anchors[str(folder/name)]=sha(folder/name)
        for p,h in s.get('sources',{}).items():check(p,h)
        for kind in ['files','checkpoints']:
            for p,h in s.get(kind,{}).items():check(folder/p,h)
        return s
    data=Path(cfg['dataset']);verify(data,'release_manifest.json')
    v3cache=Path(cfg['cache_root']);verify(v3cache,'seal.json')
    cache=verify(v4['cache_root'],'seal.json');check(v3cache/'seal.json',cache['anchor_cache_seal_sha']);check(Path(cfg['training_root'])/'soft3_31.pt',cache['coarse_model_sha'])
    train=verify(v4['training_root'],'training_seal.json');check(Path(v4['cache_root'])/'seal.json',train['cache_sha'])
    parent=Path(cfg['comparison_root']);verify(parent,'prediction_seal.json');verify(root,'prediction_seal.json')
    control=Path('results/material_continuous_retention_control_v4');verify(control,'prediction_seal.json')
    check(root/'prediction_seal.json',load(control/'frozen_protocol.json')['main_prediction_seal_sha'])
    p=load(root/'frozen_protocol.json');check(parent/'prediction_seal.json',p['parent_prediction_seal_sha']);check(Path(v4['training_root'])/'training_seal.json',p['training_seal_sha']);check(data/'release_manifest.json',p['dataset_release_sha'])
    for r in [parent,root]:
        a=load(r/'scoring_audit.json');source=parent/'scorer_source_at_initial_scoring.py' if r==parent else Path('scripts/score_material_continuous_v4.py');check(source,a['source_sha'])
        for path,h in a['hashes'].items():check(r/path,h)
    check(control/'prediction_seal.json',load(root/'scoring_audit.json')['prescore_control_seal_sha'])
    for r in ['results/material_joint_descriptor_training_v1','results/material_joint_descriptor_training_v2','results/material_correspondence_training_v3']:verify(r,'training_seal.json')
    for r in ['results/textured_material_comparison_v1','results/material_joint_descriptor_comparison_v1','results/material_joint_descriptor_comparison_v2']:
        a=load(Path(r)/'scoring_audit.json')
        for path,h in a['hashes'].items():check(Path(r)/path,h)
    for name in p['scenes']:
        for v in cfg['variants']:check(root/name/f'{v}_baseline.npz',sha(parent/name/f'{v}_ray.npz'))
    old=Path('results/surface_measure_first_return_v1/scores.json');check(old,'196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4')
    d=load(root/'mode_diagnostics.json');check('scripts/diagnose_material_continuous_v4.py',d['source_sha'])
    d=load(root/'selection_diagnostic.json');check('scripts/diagnose_material_continuous_selection_v4.py',d['source_sha'])
    d=load(Path(v4['training_root'])/'train_replay_diagnostic.json');check('scripts/diagnose_material_continuous_train_v4.py',d['source_sha'])
    report=load(root/'report_manifest.json');check('scripts/report_material_continuous_v4.py',report['source_sha'])
    for k in ['sources','outputs']:
        for path,h in report[k].items():check(path,h)
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,all_scientific_sources_and_data_unchanged=True,
        all_new_predictions_and_models_sealed=True,baseline_byte_identical=True,old_192_sha=sha(old),
        figures_visually_inspected=['continuous_task_and_oracle.png','continuous_uncovered_candidate_recall.png'],
        research_artifacts={p:sha(p) for p in ['research/material_continuous_correspondence_protocol_v4_2026-10-26.md','research/material_continuous_correspondence_prior_art_v4_2026-10-26.md','research/material_joint_identity_material_next_v5_2026-10-26.md']},
        no_independent_confirmation=True,no_new_real_data_accuracy=True,seal_hashes=anchors,source_sha=sha(__file__)))
    print('V4 COMPLETION INTEGRITY',checks,'PASS',flush=True)


if __name__=='__main__':main()
