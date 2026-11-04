"""Outcome-selected old scene155 objective-vs-optimization diagnostic."""
import json
from pathlib import Path
import numpy as np
from scripts.analyze_partial_surface import load_input
from scripts.shared_pose_change_v3 import robust_energy
from scripts.surface_motion_v2 import context,visibility,residual


def main():
    out=Path('results/shared_pose_change_v3');manifest=json.loads((out/'manifest.json').read_text());cfg=json.loads((out/'frozen_config.json').read_text());mcfg=json.loads((out/'motion_config.json').read_text());rows=json.loads((out/'scene155_trace.json').read_text());source=manifest['base_predictions'][str(rows[0]['scene'])]['input_path'];obs=load_input(Path(source));scale=.03/np.sqrt(mcfg['current_max_points']);comparison=[]
    for row in rows:
        T=np.array(row['T']);A=np.array(row['prediction_assignment']);delta=np.array(row['delta']);pieces=[]
        for i,j in enumerate(A):
            c=context(obs,i,int(j),mcfg);selected=visibility(c,obs,T,delta[i],mcfg,True);data=residual(c,obs,T,delta[i],int(j),selected,mcfg,'visible_bidirectional');reg=mcfg['regularization']*delta[i]
            value=robust_energy(np.r_[data,reg],scale)
            pieces.append(dict(asset=int(j),changed=bool(row['prediction_changed'][i]),truth_changed=bool(row['truth_changed'][i]),data_regularization_energy=value,change_penalty=cfg['motion_penalty_m2'] if row['prediction_changed'][i] else 0,visible_model_samples=int(selected.sum())))
        total=sum(p['data_regularization_energy']+p['change_penalty'] for p in pieces)
        comparison.append(dict(method=row['method'],reported_pose_translation_error_m=row['translation_error_m'],reported_pose_rotation_error_deg=row['rotation_error_deg'],recomputed_common_energy=total,components=pieces))
    joint=next(c for c in comparison if c['method']=='joint_profile');control=next(c for c in comparison if c['method']=='ray_gate_bidirectional')
    result=dict(scope='Posthoc single diagnostic case; common objective recomputed with each final visibility, not a global optimality proof',old_scene=155,comparison=comparison,wrong_joint_solution_has_lower_energy_than_correct_control=joint['recomputed_common_energy']<control['recomputed_common_energy'],caution='If true, better optimization of this objective alone cannot prefer the correct-control candidate over this wrong solution. It does not prove all dataset errors have this cause.')
    (out/'objective_failure_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
