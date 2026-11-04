"""Full sealed v5 -> independent no-GT audit -> task scoring -> diagnostics."""
from pathlib import Path
import subprocess,sys
from scripts.run_rigid_identity_bounds_v1 import read,sha


def main():
    root=Path('results/first_return_gamma_material_v5');seal=read(root/'prediction_seal.json');assert seal['complete'] and seal['scenes']==8
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    assert not Path('results/first_return_gamma_material_v4/evaluation_sources.json').exists()
    for module,marker,key in [('audit_first_return_gamma_material_v5','audit.json','auditor_sha256'),('score_first_return_gamma_material_v5','evaluation_sources.json','script_sha256'),('diagnose_first_return_gamma_material_v5','mechanism_summary.json','script_sha256'),('report_first_return_gamma_material_v5','report_sources.json','script_sha256')]:
        if not (root/marker).exists() or read(root/marker).get(key)!=sha(Path('scripts')/(module+'.py')):subprocess.run([sys.executable,'-m','scripts.'+module],check=True)
        assert read(root/'audit.json')['passed']
    print('FULL_V5_CHAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
