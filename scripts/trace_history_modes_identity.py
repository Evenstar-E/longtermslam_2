import hashlib,json
from pathlib import Path
import numpy as np
from scripts.history_modes_v6 import infer,clusters

def main():
    base=Path('results/history_modes_v1');plan=json.loads((base/'frozen_config.json').read_text());cfg=json.loads(Path(plan['base_algorithm']).read_text());sampler=json.loads(Path(plan['sampler']).read_text());path=base/'cases'/'2_0084.npz';a=np.load(path);obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']};p=infer(obs,cfg['algorithm'],sampler,plan['modes'],'v6_modes',plan['seed_base']+84+400000);modes=[(j,g) for j in range(24) for g in clusters(obs['history_positions'][:,j],plan['modes']['cluster_radius_m'])];ki=p['solutions'][p['selected_index']]['mode'];records=[]
    for i in np.flatnonzero(a['truth_changed']&(p['assignment']!=a['truth_assignment'])):
        j,g=modes[int(ki[i])];records.append(dict(current_index=int(i),true_physical_id=int(a['truth_assignment'][i]),reported_nominal_id=int(p['assignment'][i]),reported_changed=bool(p['changed'][i]),residual_m=float(p['residual_m'][i]),historical_mode_frames=g,historical_mode_physical_ids=a['truth_provenance'][g,j].tolist()))
    result=dict(base_scene=84,stress='majority_mix',selection='Outcome-selected identity-failure diagnostic,not representative efficacy sample',records=records,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(base/'identity_failure_trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
