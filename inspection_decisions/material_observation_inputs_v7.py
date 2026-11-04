"""Public-only observation subset adapter. No capture groups or private labels."""
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_observation


def load_epochs(cfg,condition,variant):
    dest=Path(cfg['dataset'])/cfg['scene'];records=json.loads((dest/'frames.json').read_text(encoding='utf-8'));out=[];ids=[]
    for e in range(3):
        rr=[r for r in records if r['epoch']==e and (condition=='full24' or r['kind']=='overview')];order=np.random.default_rng(cfg['seed']+e).permutation(len(rr));rr=[rr[i] for i in order]
        out.append([load_observation(dest/r['observation'],variant) for r in rr]);ids.append([r['index'] for r in rr])
    return out,ids
