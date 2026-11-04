"""Load the fixed published prior. This module never reads evaluator IDs."""
import os,sys,random
from pathlib import Path
import numpy as np

def load_solver(project,cfg):
    project=Path(project).resolve();port=project/'third_party/LivingScenes_cpu_port_v1'
    sys.path.insert(0,str(port))
    import torch,yaml
    torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    from lib_more.more_solver import More_Solver
    solver_cfg=yaml.safe_load((port/'configs/more_3rscan.yaml').read_text(encoding='utf-8'))
    solver_cfg['shape_priors']['ckpt_dir']=str(project/'third_party/LivingScenes/weights')
    old=Path.cwd();os.chdir(port)
    try:solver=More_Solver(solver_cfg)
    finally:os.chdir(old)
    return solver
