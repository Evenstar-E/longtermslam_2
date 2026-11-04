"""Actual-patch numerical port diagnostic, not a data-set accuracy experiment."""
import importlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.livingscenes_cpu_runtime_v1 import load_solver

cfg=json.loads(Path('configs/livingscenes_real_v1.json').read_text(encoding='utf-8'));solver=load_solver(Path.cwd(),cfg)
import torch
from pytorch3d.ops import sample_farthest_points
from collections import namedtuple
root=Path('results/livingscenes_cpu_port_v1/model_numerical_checks');root.mkdir(exist_ok=False)
with np.load('data/real_cross_session_v1/prepared_v2/public/scene_a/0/o000.npz') as z:x=torch.from_numpy(z['xyz'].astype(np.float32))[None]
with torch.no_grad():x,_=sample_farthest_points(x,K=1024)
R=torch.tensor(Rotation.from_rotvec([.17,-.11,.23]).as_matrix(),dtype=torch.float32);t=torch.tensor([.8,-.4,.3])
with torch.no_grad():
    a=solver.model.encode(x.transpose(1,2));b=solver.model.encode((x@R.T+t).transpose(1,2))
    record={'invariant_cosine':float(torch.nn.functional.cosine_similarity(a['z_inv'],b['z_inv']).item()),
        'equivariant_feature_relative_error':float((b['z_so3']-a['z_so3']@R.T).norm()/a['z_so3'].norm()),
        'scale_relative_error':float(((a['s']-b['s'])/a['s']).abs().item()),
        'center_equivariance_error_m':float((b['t']-(a['t']@R.T+t)).norm())}
    # Independent, direct distance enumeration on the same points validates the CPU KNN path.
    module=importlib.import_module('lib_shape_prior.core.lib.vec_sim3.vec_dgcnn_atten');old=module.knn_points
    KNN=namedtuple('_KNN',['dists','idx','knn'])
    def brute(p1,p2,K=1,return_nn=False,**unused):
        # Query chunks bound temporary memory, retaining direct squared-distance summation.
        values=[];indices=[]
        for start in range(0,p1.shape[1],32):
            d=((p1[:,start:start+32,None]-p2[:,None])**2).sum(-1)
            v,i=d.topk(K,largest=False);values.append(v);indices.append(i)
        v=torch.cat(values,1);i=torch.cat(indices,1)
        n=p2[torch.arange(len(p2))[:,None,None],i] if return_nn else None
        return KNN(v,i,n)
    module.knn_points=brute
    try:c=solver.model.encode(x.transpose(1,2))
    finally:module.knn_points=old
    record['direct_knn_descriptor_cosine']=float(torch.nn.functional.cosine_similarity(a['z_inv'],c['z_inv']).item())
    record['direct_knn_max_code_difference']={k:float((a[k]-c[k]).abs().max()) for k in a}
record['scope']='Single actual patch, transformed in software for numerical diagnostics; not independent real scene or performance claim.'
(root/'checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
