"""Final source/data/model/prediction/report integrity; no experiment reruns."""
import csv,json
from pathlib import Path
from scripts.run_textured_material_comparison_v1 import save,sha

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    cfg=load('configs/material_joint_shared_v5.json');root=Path(cfg['output']);p=load(root/'frozen_protocol.json');checks=0;anchors={}
    def check(file,expected):
        nonlocal checks
        assert sha(file)==expected,str(file);checks+=1
    def verify(folder,name):
        folder=Path(folder);s=load(folder/name);assert s['complete'];anchors[str(folder/name)]=sha(folder/name)
        for file,h in s.get('sources',{}).items():check(file,h)
        for k in ['files','checkpoints']:
            for file,h in s.get(k,{}).items():check(folder/file,h)
        return s
    data=Path(p['base_config']['dataset']);verify(data,'release_manifest.json');verify(root,'prediction_seal.json')
    parent=Path(p['v4']['comparison_root']);verify(parent,'prediction_seal.json');check(parent/'prediction_seal.json',p['parent_prediction_seal_sha'])
    verify(p['v4']['training_root'],'training_seal.json');check(p['feature_checkpoint'],p['feature_checkpoint_sha']);check(data/'release_manifest.json',p['dataset_release_sha'])
    score=load(root/'scoring_audit.json');check('scripts/score_material_joint_shared_v5.py',score['source_sha'])
    for file,h in score['hashes'].items():check(root/file,h)
    for name,source in [('objective_reconstruction_audit.json','scripts/audit_material_joint_shared_objectives_v5.py'),('real_search_regression.json','scripts/check_material_joint_real_regression_v5.py')]:
        s=load(root/name);assert s['complete'];check(source,s['source_sha'])
    for name in p['scenes']:
        for v in p['base_config']['variants']:check(root/name/f'{v}_union_pools.npz',sha(parent/name/f'{v}_union_pools.npz'))
    for folder in ['results/textured_material_comparison_v1','results/material_joint_descriptor_comparison_v1','results/material_joint_descriptor_comparison_v2',p['base_config']['comparison_root'],str(parent)]:
        a=load(Path(folder)/'scoring_audit.json')
        for file,h in a['hashes'].items():check(Path(folder)/file,h)
    old=Path('results/surface_measure_first_return_v1/scores.json');check(old,'196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4')
    report=load(root/'report_manifest.json');check('scripts/report_material_joint_shared_v5.py',report['source_sha'])
    for k in ['sources','outputs']:
        for file,h in report[k].items():check(file,h)
    with (root/'task_scores.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for v in p['base_config']['variants']:
        sets=[{(r['scene'],r['asset']) for r in rows if r['variant']==v and r['method']==m and int(r['qualified'])} for m in p['methods']]
        assert all(a==sets[0] for a in sets);checks+=4
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,all_new_sources_data_models_predictions_verified=True,
        unchanged_parent_pose_pools=True,all_five_qualified_sets_identical=True,original_192_sha=sha(old),
        figures_visually_inspected=['joint_shared_task_comparison.png','shared_factor_scale.png'],no_new_independent_confirmation=True,
        seal_hashes=anchors,source_sha=sha(__file__),research_artifacts={file:sha(file) for file in ['research/material_joint_shared_protocol_v5_2026-10-26.md','research/material_correspondence_modes_next_v6_2026-10-26.md']}))
    print('V5 COMPLETION INTEGRITY',checks,'PASS',flush=True)

if __name__=='__main__':main()
