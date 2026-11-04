"""Wait for complete sealed predictions, then score/diagnose/report once."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
import time
from pathlib import Path


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));root=Path(v4['comparison_root']);start=time.monotonic();last=0
    if (root/'scoring_audit.json').exists():raise RuntimeError('Already scored; refusing to overwrite sealed evidence')
    while True:
        try:complete=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
        except (FileNotFoundError,json.JSONDecodeError):complete=False
        if complete:break
        if time.monotonic()-start>3600:raise TimeoutError('No completed prediction seal within one hour; inference was not killed')
        if time.monotonic()-last>60:
            statuses={p.parent.name:json.loads(p.read_text(encoding='utf-8'))['message'] for p in root.glob('*/progress.json')}
            print('V4 WAITING FOR COMPLETE SEAL',statuses,flush=True);last=time.monotonic()
        time.sleep(2)
    from scripts.score_material_continuous_v4 import main as score
    from scripts.run_material_continuous_retention_control_v4 import main as retained_control
    from scripts.diagnose_material_continuous_v4 import main as diagnose
    from scripts.report_material_continuous_v4 import main as report
    retained_control();score();diagnose();report();print('V4 SCORE DIAGNOSTIC REPORT COMPLETE; visual QA and final audit pending',flush=True)


if __name__=='__main__':main()
