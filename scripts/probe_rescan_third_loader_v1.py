"""Bounded line-buffered probe; no inference result or score produced."""
import subprocess
from pathlib import Path
from scripts.run_rescan_third_period_v1 import linux, save, sha


def main():
    out=Path('results/rescan_third_loader_probe_v1');out.mkdir(exist_ok=False)
    binary=Path('third_party/Rescan_memory_port_v1/bin/pose_proposal')
    db=Path('data/rescan_protected_archive_v1/scene_a/prediction.rsdb')
    current=Path('data/rescan_third_capture_v1/public/scene_a/capture2.ply')
    command=['wsl','-d','Ubuntu-20.04','--exec','env','OMP_NUM_THREADS=1','timeout','--kill-after=5','20',
             'stdbuf','-oL','-eL',linux(binary),linux(db),linux(current),linux(out/'probe.rsdb'),'-v']
    with (out/'probe.log').open('xb') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=40)
    save(out/'probe.json',{'returncode':r.returncode,'timeout_s':20,'inference_score':None,'inputs':{str(p):sha(p) for p in [binary,db,current]},'source_sha':sha(__file__)})
    print('probe exit',r.returncode)


if __name__=='__main__':main()
