"""Resumable: all8 seal -> no-GT audit -> formal score -> mechanism diagnostic."""
from pathlib import Path
import subprocess,sys
from scripts.run_rigid_identity_bounds_v1 import read,sha


def main():
    root=Path('results/sparse_material_evidence_v1');assert all((root/f'shard_{j}'/'prediction_seal.json').exists() for j in range(3))
    if not (root/'prediction_seal.json').exists():subprocess.run([sys.executable,'-m','scripts.merge_sparse_material_evidence_v1'],check=True)
    assert read(root/'prediction_seal.json')['scenes']==8
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    for module,marker,key in [('audit_sparse_material_evidence_v1','audit.json','auditor_sha256'),('score_sparse_material_evidence_v1','evaluation_sources.json','script_sha256'),('diagnose_sparse_material_evidence_v1','diagnostic_summary.json','script_sha256')]:
        path=root/marker
        if not path.exists() or read(path).get(key)!=sha(Path('scripts')/(module+'.py')):subprocess.run([sys.executable,'-m','scripts.'+module],check=True)
        assert read(root/'audit.json')['passed']
    print('FULL_EVIDENCE_CHAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
