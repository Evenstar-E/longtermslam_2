"""Two-step synthetic API check only; never counted as real-data performance."""
import json,time
from pathlib import Path
import numpy as np
from inspection_decisions.livingscenes_cpu_runtime_v1 import load_solver
root=Path('results/livingscenes_cpu_port_v1/pose_api_check');root.mkdir(exist_ok=False)
cfg=json.loads(Path('configs/livingscenes_real_v1.json').read_text(encoding='utf-8'));solver=load_solver(Path.cwd(),cfg)
import torch
solver.cfg['registration']['n_steps']=2
with np.load('data/real_cross_session_v1/prepared_v2/public/scene_a/0/o000.npz') as z:a=torch.tensor(z['xyz'].astype(np.float32))[None]
b=a+torch.tensor([[[.03,-.02,.01]]])
start=time.perf_counter();R,t=solver._solve_pairwise_registration(a,b,optim=True)
record={'scope':'Synthetic two-step software API check, no real experimental motion sample','elapsed_s':time.perf_counter()-start,
    'R':R.detach().numpy().tolist(),'t':t.detach().numpy().tolist(),'finite':bool(torch.isfinite(R).all() and torch.isfinite(t).all()),
    'rotation_orthogonal':bool(torch.allclose(R@R.transpose(-1,-2),torch.eye(3)[None],atol=1e-4))}
(root/'check.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
assert record['finite'] and record['rotation_orthogonal']
