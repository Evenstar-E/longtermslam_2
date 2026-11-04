"""Create versioned config/evaluator; keep all sealed V7 sources unchanged."""
import json
from pathlib import Path
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=json.loads(Path('configs/material_partial_track_transport_v7.json').read_text(encoding='utf-8'));cfg['output']='results/material_partial_track_fixed_measure_v8';cfg['contract']='TRAIN correction: fixed observed node measure, identical gains/capacity/partition/SE3 rules; not a new theory and not independent confirmation.';save(Path('configs/material_partial_track_fixed_measure_v8.json'),cfg)
    source=Path('scripts/score_material_partial_track_transport_v7.py').read_text(encoding='utf-8').replace('configs/material_partial_track_transport_v7.json','configs/material_partial_track_fixed_measure_v8.json');source=source.replace('from scripts.run_material_partial_track_transport_v7 import METHODS,setup','from scripts.run_material_partial_track_transport_v7 import METHODS,setup\nfrom inspection_decisions.material_partial_track_fixed_measure_v8 import FixedMeasureTransport as PairTransport');Path('scripts/score_material_partial_track_fixed_measure_v8.py').write_text(source,encoding='utf-8')
    d=json.loads(Path('results/material_partial_track_transport_v7/mechanism_diagnosis.json').read_text(encoding='utf-8'));rows=[]
    for r in d['loss_cases']:
        a,b=r['good_feasible'],r['bad_selected'];rows.append(dict(variant=r['variant'],asset=r['asset'],same_identity=r['same_identity'],bad_has_less_raw_gain=b['shared_gain']<a['shared_gain'],bad_has_more_free_conflict=b['free']>a['free'],bad_has_smaller_denominator=b['Z']<a['Z'],bad_wins_original=r['new_energy_change']<0))
    save(Path('results/material_fixed_measure_counterexample_v8.json'),dict(rows=rows,strict_domination_cases=sum(r['same_identity'] and r['bad_has_less_raw_gain'] and r['bad_has_more_free_conflict'] and r['bad_wins_original'] for r in rows),proof='For fixed positive Z0, less raw gain AND more free conflict always imply higher energy. Pose-dependent Z violates this implication. This is a formula counterexample, not task efficacy or novelty.'))

if __name__=='__main__':main()
