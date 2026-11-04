"""Same full task comparison after the documented TRAIN energy counterexample."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from inspection_decisions.material_partial_track_fixed_measure_v8 import FixedMeasureTransport
from scripts import run_material_partial_track_transport_v7 as original
from scripts.run_textured_material_comparison_v1 import save,sha

def job(cfg,v):
    # Explicit dependency injection; sealed source files are never changed.
    original.PairTransport=FixedMeasureTransport
    return original.job(cfg,v)

def main():
    cp=Path('configs/material_partial_track_fixed_measure_v8.json');cfg=json.loads(cp.read_text(encoding='utf-8'));previous=Path('results/material_partial_track_transport_v7');seal=json.loads((previous/'prediction_seal.json').read_text(encoding='utf-8'))
    for p,h in seal['sources'].items():assert sha(p)==h
    for p,h in seal['files'].items():assert sha(previous/p)==h
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);paths=[__file__,str(cp),'inspection_decisions/material_partial_track_fixed_measure_v8.py','inspection_decisions/material_partial_track_transport_v7.py','scripts/run_material_partial_track_transport_v7.py',cfg['input_config'],cfg['registration_config']];sources={p:sha(p) for p in paths};start=time.perf_counter()
    save(root/'frozen_protocol.json',dict(config=cfg,methods=original.METHODS,sources=sources,previous_seal_sha=sha(previous/'prediction_seal.json'),diagnosis_sha=sha(previous/'mechanism_diagnosis.json'),private_labels_read=False,scope='Outcome-informed TRAIN repair, not independent confirmation. Fixed observation count denominator; all remaining gains/partition/SE3 settings unchanged. Same shared candidate generation rules, not identical realized endpoint or identity pools across versions. Four methods within version have same inputs/final candidates, not equal FLOPs.'))
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in as_completed([pool.submit(job,cfg,v) for v in cfg['variants']]):f.result()
    for p,h in sources.items():assert sha(p)==h
    save(root/'prediction_seal.json',dict(complete=True,sources=sources,elapsed_s=time.perf_counter()-start,private_labels_read=False,files={str(p.relative_to(root)):sha(p) for p in root.glob('*.npz')}));print('FIXED MEASURE SEALED',flush=True)

if __name__=='__main__':main()
