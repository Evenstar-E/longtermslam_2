"""Wait for all frozen arms, then evaluate once; incomplete runs never become scores."""
import subprocess
import sys
import time
from pathlib import Path
from scripts.prepare_rescan_three_period_v1 import run_root
from scripts.run_rescan_third_period_v1 import save
import json


def main():
    base=Path('results/rescan_third_period_progress_v1');started=time.monotonic()
    while True:
        pending=[]
        for scene in ['scene_a','scene_c']:
            for mode in ['initial_archive','inherited_unfused','inherited_protected']:
                root=run_root(scene,mode,True)
                if (root/'prediction_seal.json').exists():continue
                if (root/'run.json').exists():
                    data=json.loads((root/'run.json').read_text(encoding='utf-8'))
                    if any(r['returncode']!=0 for r in data['records']):
                        raise RuntimeError(f'No scoring: incomplete arm {root}')
                pending.append(f'{scene}/{mode}')
        save(base/'completion_monitor.json',{'pending':pending,'elapsed_s':time.monotonic()-started,'evaluation_started':False})
        if not pending:break
        if time.monotonic()-started>3700:raise TimeoutError('Waiting for complete inference exceeded declared limit')
        time.sleep(5)
    for module in ['score_rescan_third_period_v1','diagnose_rescan_third_period_v1']:
        with (base/f'{module}.log').open('xb') as log:
            r=subprocess.run([sys.executable,'-m','scripts.'+module,'--progress'],stdout=log,stderr=subprocess.STDOUT)
        print(module,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(module)
    save(base/'completion_monitor.json',{'pending':[],'evaluation_complete':True})


if __name__=='__main__':main()
