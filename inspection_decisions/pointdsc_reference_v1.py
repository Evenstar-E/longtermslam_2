"""Load unchanged pinned author modules, weights and config on CPU.

No model port, algorithm rewrites or global package installation. Temporary
module aliases are restored after import to avoid repository namespace clashes.
"""
import importlib.util,json,sys,types
from pathlib import Path
import numpy as np
import torch


def load_reference():
    root=Path('third_party/PointDSC_official_v1');names=['utils','utils.SE3','models','models.common','models.PointDSC'];previous={name:sys.modules.get(name) for name in names}
    try:
        for package in ['utils','models']:
            module=types.ModuleType(package);module.__path__=[str(root/package)];sys.modules[package]=module
        for name,relative in [('utils.SE3','utils/SE3.py'),('models.common','models/common.py'),('models.PointDSC','models/PointDSC.py')]:
            spec=importlib.util.spec_from_file_location(name,root/relative);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        cls=sys.modules['models.PointDSC'].PointDSC
    finally:
        for name,old in previous.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old
    base=root/'snapshot/PointDSC_3DMatch_release';cfg=json.loads((base/'config.json').read_text(encoding='utf-8'));model=cls(in_dim=cfg['in_dim'],num_layers=cfg['num_layers'],num_channels=cfg['num_channels'],num_iterations=cfg['num_iterations'],ratio=cfg['ratio'],sigma_d=cfg['sigma_d'],k=cfg['k'],nms_radius=cfg['inlier_threshold']).cpu()
    # The official demo uses strict=False. Released weights retain an unused
    # legacy gamma although its only use is commented out in the same commit.
    report=model.load_state_dict(torch.load(base/'models/model_best.pkl',map_location='cpu',weights_only=True),strict=False)
    assert report.missing_keys==[] and report.unexpected_keys==['gamma'],report
    model._reference_load_report=dict(missing=report.missing_keys,unexpected=report.unexpected_keys)
    return model.eval(),cfg


@torch.inference_mode()
def estimate(model,x,y):
    x,y=np.asarray(x,np.float32),np.asarray(y,np.float32);corr=np.c_[x,y];corr-=corr.mean(0);data=dict(corr_pos=torch.from_numpy(corr)[None],src_keypts=torch.from_numpy(x)[None],tgt_keypts=torch.from_numpy(y)[None],testing=True);out=model(data);return out['final_trans'][0].cpu().numpy(),out['final_labels'][0].cpu().numpy()
