"""CPU adapter for pinned official SuperPoint and LightGlue sources/weights.

Original algorithm sources remain unchanged on disk. Two SuperPoint utility
imports are replaced locally (grayscale and configuration holder); no resizing
is performed. All official network layers/forward computations are retained.
"""
import importlib.util
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F


ROOT=Path('third_party/material_local_matching_v1')


def rgb_to_grayscale(image):
    return (image*image.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)


class Extractor(nn.Module):
    def __init__(self,**conf):
        super().__init__();self.conf=SimpleNamespace(**{'origin':'center',**self.default_conf,**conf})

    @property
    def coords_offset(self):
        return .5 if self.conf.origin=='corner' else 0.


def local_weights(url,*args,**kwargs):
    name='superpoint_lightglue.pth' if 'lightglue' in url.rsplit('/',1)[-1] else 'superpoint_v1.pth'
    return torch.load(ROOT/name,map_location='cpu',weights_only=True)


class PretrainedLocal:
    def __init__(self):
        original=(ROOT/'superpoint.py').read_text(encoding='utf-8')
        port=original.replace('from kornia.color import rgb_to_grayscale','from inspection_decisions.material_pretrained_v1 import rgb_to_grayscale')
        port=port.replace('from .utils import Extractor','from inspection_decisions.material_pretrained_v1 import Extractor')
        module=ModuleType('material_superpoint_official_port')
        exec(compile(port,str(ROOT/'superpoint.py'),'exec'),module.__dict__)
        self.sp_module=module
        spec=importlib.util.spec_from_file_location('material_lightglue_official',ROOT/'lightglue.py')
        lg=importlib.util.module_from_spec(spec);spec.loader.exec_module(lg)
        with patch('torch.hub.load_state_dict_from_url',local_weights):
            self.extractor=module.SuperPoint(max_num_keypoints=1024).eval()
            self.matcher=lg.LightGlue(features='superpoint',flash=False,depth_confidence=-1,width_confidence=-1).eval()
        self.last_dense=None
        self.extractor.convDb.register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):
        self.last_dense=F.normalize(output,p=2,dim=1)

    @torch.inference_mode()
    def descriptors(self,frame,uv):
        image=torch.from_numpy(frame['rgb'].transpose(2,0,1).copy()).float()[None]/255.
        # Official SuperPoint requires image dimensions divisible by eight.
        h,w=image.shape[-2:]
        image=F.pad(image,(0,(-w)%8,0,(-h)%8),mode='replicate')
        self.extractor({'image':image})
        points=torch.as_tensor(uv,dtype=torch.float32)[None]
        out=self.sp_module.sample_descriptors(points,self.last_dense,8)[0].T
        return out.numpy()

    @torch.inference_mode()
    def match(self,a,b):
        if not len(a['uv']) or not len(b['uv']):return []
        def item(x):
            return dict(keypoints=torch.from_numpy(x['uv']).float()[None],
                descriptors=torch.from_numpy(x['desc']).float()[None],
                image_size=torch.tensor([x['size']],dtype=torch.float32))
        out=self.matcher(dict(image0=item(a),image1=item(b)))
        return out['matches'][0].numpy()
