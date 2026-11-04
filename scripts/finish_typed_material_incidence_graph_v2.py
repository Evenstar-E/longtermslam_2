"""Run the fixed completion chain; evaluation is gated on full seal and audit."""
import subprocess,sys
from pathlib import Path
from scripts.run_rigid_identity_bounds_v1 import read,sha


def main():
    root=Path('results/typed_material_incidence_graph_v2')
    assert all((root/f'shard_{j}'/'prediction_seal.json').exists() for j in range(3))
    if not (root/'prediction_seal.json').exists():subprocess.run([sys.executable,'-m','scripts.merge_typed_material_incidence_graph_v2'],check=True)
    seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h
    for module,marker,key in [('audit_typed_material_incidence_graph_v2','audit.json','auditor_sha256'),('score_typed_material_incidence_graph_v2','evaluation_sources.json','evaluator_sha256'),('diagnose_typed_material_incidence_graph_v2','source_association_sources.json','script_sha256')]:
        path=root/marker;current=sha(Path('scripts')/(module+'.py'))
        if not path.exists() or read(path).get(key)!=current:subprocess.run([sys.executable,'-m','scripts.'+module],check=True)
        assert read(root/'audit.json')['passed']
    print('COMPLETE: full predictions, no-GT audit, evaluation and source diagnostic',flush=True)


if __name__=='__main__':main()
