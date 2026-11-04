"""Verify frozen data/code/models/predictions without rerunning experiments."""
import hashlib
import json
from pathlib import Path

import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel
from scripts.train_material_appearance_v3 import physical_hash


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks=0; anchors={}
    def check(p,expected):
        nonlocal checks
        assert sha(p)==expected, str(p)
        checks+=1
    def seal(root,name):
        r=Path(root);s=load(r/name)
        assert s['complete'];anchors[str(r/name)]=sha(r/name)
        for p,h in s.get('sources',{}).items():check(p,h)
        for kind in ['files','checkpoints']:
            for p,h in s.get(kind,{}).items():check(r/p,h)
        return s
    release=Path('data/textured_material_v1');s=seal(release,'release_manifest.json')
    assert len(s['files'])==1031
    cache=Path('data/material_correspondence_train_cache_v3');seal(cache,'seal.json')
    train=Path('results/material_correspondence_training_v3');tr=seal(train,'training_seal.json')
    check(cache/'seal.json',tr['cache_sha']);check('results/material_joint_descriptor_training_v1/contrastive_31.pt',tr['warm_start_sha'])
    ap=Path('results/material_appearance_training_v3');app=seal(ap,'training_seal.json');check('scripts/train_material_appearance_v3.py',app['source_sha'])
    for seed in [31,47,73]:
        models=[]
        for r in [train,ap]:
            ck=torch.load(r/f'soft3_{seed}.pt',weights_only=True,map_location='cpu');m=CorrespondenceModel();m.load_state_dict(ck['state_dict']);models.append(m)
        assert physical_hash(models[0])==physical_hash(models[1]);checks+=1
    mainroot=Path('results/material_correspondence_comparison_v3');reproot=Path('results/material_correspondence_replacement_v3')
    for r in [mainroot,reproot]:
        seal(r,'prediction_seal.json');audit=load(r/'scoring_audit.json')
        source=mainroot/'scorer_source_at_initial_scoring.py' if r==mainroot else Path('scripts/score_material_correspondence_v3.py')
        check(source,audit['source_sha'])
        for p,h in audit['hashes'].items():check(r/p,h)
        assert audit['old_pose_retention_checked'] and audit['all_refinement_paths_monotone'];checks+=2
        proto=load(r/'frozen_protocol.json')
        if 'dataset_release_sha' in proto:check(release/'release_manifest.json',proto['dataset_release_sha'])
    p=load(mainroot/'frozen_protocol.json')
    check(train/'training_seal.json',p['training_seal_sha'])
    check('results/textured_material_comparison_v1/prediction_seal.json',p['legacy_prediction_seal_sha'])
    check('results/material_joint_descriptor_comparison_v1/prediction_seal.json',p['official_prediction_seal_sha'])
    for r in ['results/material_joint_descriptor_training_v1','results/material_joint_descriptor_training_v2']:seal(r,'training_seal.json')
    for root in ['results/textured_material_comparison_v1','results/material_joint_descriptor_comparison_v1','results/material_joint_descriptor_comparison_v2']:
        r=Path(root);seal(r,'prediction_seal.json');a=load(r/'scoring_audit.json')
        for p,h in a['hashes'].items():check(r/p,h)
    for pool in mainroot.glob('*/*_union_pools.npz'):check(reproot/pool.relative_to(mainroot),sha(pool))
    for pred in mainroot.glob('*/*_ordinary*.npz'):check(reproot/pred.relative_to(mainroot),sha(pred))
    old=Path('results/surface_measure_first_return_v1/scores.json')
    check(old,'196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4')
    diagnostic=load(mainroot/'repair_diagnostics.json');check('scripts/evaluate_material_correspondence_repairs_v3.py',diagnostic['source_sha'])
    report=load(mainroot/'report_manifest.json')
    for k in ['source_hashes','outputs']:
        for p,h in report[k].items():check(p,h)
    check('scripts/report_material_correspondence_v3.py',report['source_script_sha'])
    record=dict(complete=True,checks=checks,core_evidence_unchanged=True,source_model_data_seals_verified=True,
                original_release_files=1031,physical_weights_unchanged_by_appearance_training=True,
                replacement_pools_byte_identical=True,old_192_scores_sha=sha(old),
                main_score_source_snapshot=str(mainroot/'scorer_source_at_initial_scoring.py'),
                appearance_positive_train_pairs=load(ap/'frozen_protocol.json')['repainted_pairs'],
                figures_visually_inspected=['repair_task_comparison.png','repair_appearance_operating_points.png'],
                no_independent_confirmation=True,seal_hashes=anchors,
                audit_script_sha=sha(__file__),torch=torch.__version__)
    for r in [mainroot,reproot]:(r/'completion_integrity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,checks=checks,old_192_unchanged=True)))


if __name__=='__main__':main()
