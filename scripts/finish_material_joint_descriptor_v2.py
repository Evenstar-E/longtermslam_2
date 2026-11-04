"""Wait boundedly for both complete seals, score, diagnose, then report."""
import subprocess
import sys
import time
from pathlib import Path


def main():
    roots=[Path('results/material_joint_descriptor_comparison_v1'),Path('results/material_joint_descriptor_comparison_v2')]
    start=time.monotonic()
    while not all((p/'prediction_seal.json').exists() for p in roots):
        if time.monotonic()-start>7200:raise TimeoutError('Prediction seals not completed; no missing runs scored as zero')
        time.sleep(10)
    for root in roots:
        subprocess.run([sys.executable,'-B','-m','scripts.score_material_joint_descriptor_comparison_v1','--root',str(root)],check=True)
    for module in ['scripts.evaluate_material_descriptor_heads_v2','scripts.report_material_joint_descriptor_v2']:
        subprocess.run([sys.executable,'-B','-m',module],check=True)
    print('ALL LEARNED EXPERIMENTS SCORED AND REPORTED',flush=True)


if __name__=='__main__':main()
