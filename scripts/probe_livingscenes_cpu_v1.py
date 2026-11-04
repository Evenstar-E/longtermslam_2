"""Load all published encoder/decoder weights and execute one actual observed object."""
import os,sys,time,json
from pathlib import Path
project=Path.cwd();port=project/'third_party/LivingScenes_cpu_port_v1';sys.path.insert(0,str(port))
import torch,yaml,numpy as np
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from lib_more.more_solver import More_Solver
cfg=yaml.safe_load((port/'configs/more_3rscan.yaml').read_text(encoding='utf-8'))
cfg['shape_priors']['ckpt_dir']=str(project/'third_party/LivingScenes/weights')
os.chdir(port)
try:solver=More_Solver(cfg)
finally:os.chdir(project)
with np.load(project/'data/real_cross_session_v1/prepared_v2/public/scene_a/0/o000.npz') as z:
    x=torch.from_numpy(z['xyz'].astype(np.float32)).T[None]
start=time.perf_counter()
with torch.no_grad():code=solver.model.encode_fps(x,torch.ones((1,1,x.shape[-1]),dtype=torch.bool))
record={'torch':torch.__version__,'elapsed_s':time.perf_counter()-start,'encoder_params':sum(p.numel() for p in solver.model.encoder.parameters()),
    'decoder_params':sum(p.numel() for p in solver.model.decoder.parameters()),'code_shapes':{k:list(v.shape) for k,v in code.items()},
    'finite':all(torch.isfinite(v).all().item() for v in code.values()),'shape_only_not_performance_result':True}
Path('results/livingscenes_cpu_port_v1/probe.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
