"""Index this completed transfer experiment without modifying earlier seals."""
import hashlib,json,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=Path('results/livingscenes_real_v1')
    roots=[out,Path('results/livingscenes_native_masks_v1'),Path('results/livingscenes_cpu_port_v1'),Path('data/livingscenes_native_masks_v1'),Path('third_party/pytorch3d_python_v074')]
    sources=list(Path('scripts').glob('*livingscenes*v1.py'))
    sources += [Path('scripts/fetch_pytorch3d_cpu_sources_v1.py'),Path('inspection_decisions/livingscenes_cpu_runtime_v1.py'),Path('inspection_decisions/pytorch3d_cpu_knn_v1.py'),Path('tests/test_livingscenes_cpu_port_v1.py')]
    sources += [Path('configs/livingscenes_real_v1.json'),Path('configs/livingscenes_native_masks_v1.json'),Path('research/livingscenes_real_transfer_v1_2026-10-24.md'),Path('research/joint_identity_motion_next_contract_v1_2026-10-24.md')]
    artifacts={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file() and p.name!='final_evidence_index.json' and '__pycache__' not in p.parts}
    audit=json.loads((out/'audit.json').read_text());assert all(r['pass'] for r in audit['checks'])
    port=json.loads(Path('results/livingscenes_cpu_port_v1/port_manifest_final.json').read_text())
    for name,h in port['files_sha256'].items():assert sha(Path('third_party/LivingScenes_cpu_port_v1')/name)==h
    record={'sources_sha256':{str(p):sha(p) for p in sources},'artifacts_sha256':artifacts,
        'original_revision':subprocess.check_output(['git','-C','third_party/LivingScenes','rev-parse','HEAD'],text=True).strip(),
        'checkpoint_sha256':sha('third_party/LivingScenes/weights/checkpoint/LivingScenes_latest.pt'),
        'port_final_manifest_sha256':sha('results/livingscenes_cpu_port_v1/port_manifest_final.json'),
        'public_metadata_at_final_seal':{str(p):sha(p) for p in [Path('data/real_cross_session_v1/prepared_v2/public.json'),Path('data/livingscenes_native_masks_v1/public.json')]},
        'metadata_limitation':'Initial runner hashes public NPZ payloads but not public.json. GT-mask public.json matches previous experiment seal; native public metadata is reconstructed/verified against native prediction in audit, then hashed here. No claim of its independent pre-inference hash.',
        'scope':'CPU task-inference transfer, not original CUDA/3RScan/reconstruction benchmark. Four full400-step pairwise probes, not all-object full-pose evaluation.',
        'checks_passed':audit['passed'],'unit_tests_observed_passed':5,'new_algorithm_gain_confirmed':False,'independent_motion_GT_available':False}
    (out/'final_evidence_index.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('SEALED',len(artifacts),'artifacts',len(sources),'sources',audit['passed'],'checks')

if __name__=='__main__':main()
