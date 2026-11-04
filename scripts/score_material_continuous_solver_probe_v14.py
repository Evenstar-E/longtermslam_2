"""Post-seal score all48 public probes, without cherry-picking failures."""
import json
from pathlib import Path
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,load
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];p=json.loads((root/'public_solver_probe.json').read_text(encoding='utf-8'));assert not p['private_labels_read']
    for path,h in p['inputs'].items():assert sha(path)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);payload=load(root/'public_solver_probe.npz');rows=[]
    for m in ['continuous_pair','material_track_shared']:
        pred=load(root/f'rich_{m}.npz')
        for i,q in enumerate(anchors):
            x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);both=int(np.array_equal(pred['target_slots'][i],perm[1:,i]))
            for stage in ['initial','geometry','probe']:
                error=physical_error(*payload[f'{m}_{i}_{stage}'],x,y1,y2);rows.append(dict(method=m,asset=i,stage=stage,both_ids=both,worst_error_m=error,qualified=int(both and error<=.05)))
    write_csv(root/'solver_probe_task_metrics.csv',rows);summary=[]
    for m in ['continuous_pair','material_track_shared']:
        for stage in ['initial','geometry','probe']:
            rr=[r for r in rows if r['method']==m and r['stage']==stage];summary.append(dict(method=m,stage=stage,tasks=len(rr),qualified=sum(r['qualified'] for r in rr),correct_id_median_error_m=float(np.median([r['worst_error_m'] for r in rr if r['both_ids']]))))
    save(root/'solver_probe_scores.json',dict(summary=summary,scope='All24rich tasks per2 methods, fixed selected identities, public blind extra solver probe; never substituted for full768 main predictions. One finite-difference step is not proof of optimality or a novel algorithm.',sources={str(Path(__file__)):sha(__file__),'scripts/probe_material_continuous_solver_v14.py':sha('scripts/probe_material_continuous_solver_v14.py')},public_probe_sha=sha(root/'public_solver_probe.json')));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
