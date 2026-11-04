"""Produce a final, reviewable evidence index without overwriting earlier experiments."""
import hashlib,json,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def revision(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
def main():
    root=Path('results/real_temporal_v1')
    related=[Path('results/real_temporal_color_control_v1'),Path('results/rescan_author_v1'),Path('data/real_cross_session_v1')]
    sources=[Path('inspection_decisions/real_temporal_v1.py'),Path('configs/real_cross_session_v1.json')]
    sources += [p for p in Path('scripts').glob('*real_temporal_v1.py')]
    sources += [Path('scripts')/n for n in ['probe_public_rescan_v1.py','fetch_rescan_subset_v1.py','build_rescan_author_v1.py',
        'run_rescan_author_v1.py','score_rescan_author_v1.py','diagnose_rescan_crash_v1.py','port_rescan_memory_v1.py',
        'audit_rescan_author_v1.py','diagnose_rescan_failures_v1.py','plot_rescan_author_v1.py',
        'check_teaser_native_v1.py','posthoc_real_color_control_v1.py','seal_real_cross_session_v1.py','diagnose_real_global_frame_v1.py']]
    sources += [Path('tests/test_real_temporal_v1.py'),Path('tools/real_registration_v1/teaser_cli.cpp')]
    sources += [Path('research/real_cross_session_v1_2026-10-23.md'),Path('research/rescan_author_execution_v1_2026-10-23.md')]
    references=[{'name':n,'revision':revision(Path('third_party')/n)} for n in ['Rescan','LivingScenes','khronos_ros1_deps/teaser_plusplus']]
    artifacts={str(p):sha(p) for parent in [root]+related for p in parent.rglob('*') if p.is_file() and p.name!='final_evidence_index.json' and '__pycache__' not in str(p)}
    frozen=json.loads((root/'start_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in frozen['inference_sources'].items())
    old=json.loads(Path('results/fragment_union_joint_v1/frozen_contract.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in old['sources'].items())
    weight=Path('third_party/LivingScenes/weights/checkpoint/LivingScenes_latest.pt')
    record={'sources_sha256':{str(p):sha(p) for p in sources},'artifacts_sha256':artifacts,'references':references,
        'living_scenes_checkpoint':{'path':str(weight),'bytes':weight.stat().st_size,'sha256':sha(weight),'inference_executed':False},
        'previous_frozen_sources_unchanged':len(old['sources']),'main_inference_sources_unchanged':True,
        'metric_object_motion_ground_truth_available':False,'new_paper_algorithm_validated':False}
    (root/'final_evidence_index.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('SEALED',len(artifacts),'artifacts',len(sources),'sources')

if __name__=='__main__':main()
