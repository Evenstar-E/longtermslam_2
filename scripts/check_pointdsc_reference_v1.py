import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from inspection_decisions.pointdsc_reference_v1 import load_reference,estimate
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path('third_party/PointDSC_official_v1');manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    for f in manifest['files']:assert sha(root/f['path'])==f['sha256']
    model,cfg=load_reference();rng=np.random.default_rng(71);x=rng.uniform(-1,1,(128,3));R=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();t=np.array([.12,-.08,.04]);y=x@R.T+t;start=time.perf_counter();T,label=estimate(model,x,y);seconds=time.perf_counter()-start;error=float(np.max(np.linalg.norm(x@T[:3,:3].T+T[:3,3]-y,axis=1)));assert error<1e-4,error
    save(Path('results/pointdsc_cpu_compatibility_v1.json'),dict(passed=True,commit=manifest['commit'],weight_load=model._reference_load_report,original_algorithm_modified=False,cpu=True,parameters=sum(p.numel() for p in model.parameters()),clean_rigid_max_error_m=error,seconds=seconds,author_config=cfg,sources={__file__:sha(__file__),'inspection_decisions/pointdsc_reference_v1.py':sha('inspection_decisions/pointdsc_reference_v1.py')},official_manifest_sha=sha(root/'source_manifest.json'),scope='CPU/weight/device and clean rigid numerical compatibility check, NOT actual task benchmark, new training, novelty or comparison outcome. Author-demo strict=False, no missing active weights, only unusedgamma. Unchanged active author CPU path; inactive seed_as_center CUDA branch remains untouched.'));print('POINTDSC CPU CHECK',error,seconds)


if __name__=='__main__':main()
