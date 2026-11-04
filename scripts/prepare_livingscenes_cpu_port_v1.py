"""Create a traceable CPU task-inference port, preserving original model and solver math."""
import ast,difflib,hashlib,json,shutil
from pathlib import Path

original=Path('third_party/LivingScenes');port=Path('third_party/LivingScenes_cpu_port_v1')
shutil.copytree(original,port,ignore=shutil.ignore_patterns('.git','weights','__pycache__'))
out=Path('results/livingscenes_cpu_port_v1');out.mkdir(exist_ok=False)
patches=[]
def edit(name,old,new,count=1):
    p=port/name;text=p.read_text(encoding='utf-8');assert text.count(old)==count,(name,old,text.count(old));p.write_text(text.replace(old,new),encoding='utf-8');patches.append(name)

registry='lib_shape_prior/core/__init__.py'
registry_text=(port/registry).read_text(encoding='utf-8')
edit(registry,registry_text,'"""Inference port: training registries are not imported eagerly.\n\nThe original registry imports optional mesh extensions even for encoder-only\ninference. No training or reconstruction entry point is deployed here.\n"""\n')

edit('model_utils.py','from lib_shape_prior.core.models.utils.occnet_utils.mesh_extractor2 import Generator3D as Generator3D_MC','')
edit('model_utils.py','torch.load(osp.join(working_dir, cfg["field_pt"]))','torch.load(osp.join(working_dir, cfg["field_pt"]), map_location="cpu", weights_only=False)')
edit('model_utils.py','device = torch.device("cuda")','device = torch.device("cpu")')
edit('lib_more/more_solver.py','from lib_shape_prior.core.models.utils.occnet_utils.mesh_extractor2 import Generator3D as Generator3D_MC','')
edit('lib_more/more_solver.py','self.mesh_extractor = Generator3D_MC(**cfg["mesh_extractor"])','self.mesh_extractor = None  # mesh output is excluded from this task-inference port')
edit('lib_more/more_solver.py','from pycg import vis','')
edit('lib_more/more_solver.py','s0 = torch.tensor([1]).float().cuda()','s0 = torch.tensor([1], device=pc1.device).float()')
edit('lib_more/more_solver.py','        centroid = latent_code[\'t\'].detach().clone()',
     '        if self.mesh_extractor is None: raise NotImplementedError("Mesh extraction not deployed; this port evaluates identity and pose only")\n        centroid = latent_code[\'t\'].detach().clone()')
edit('lib_more/pose_estimation.py','from pycg import vis','')
edit('lib_more/matcher_new.py','torch.tensor(1.).cuda().float()','scores.new_tensor(1.).float()')
# Copy the original pure-Python ICP, utilities and reference FPS implementation.
official=Path('third_party/pytorch3d_python_v074')
compat=port/'pytorch3d';compat.mkdir()
for relative in ['ops/points_alignment.py','ops/utils.py','common/workaround/utils.py','structures/utils.py']:
    target=compat/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(official/'pytorch3d'/relative,target)
shutil.copyfile(official/'LICENSE',compat/'LICENSE')
for relative in ['','ops','structures','common','common/workaround']:
    (compat/relative/'__init__.py').write_text('',encoding='utf-8')
source=(official/'pytorch3d/ops/sample_farthest_points.py').read_text(encoding='utf-8')
tree=ast.parse(source)
tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='pytorch3d') and not (isinstance(node,ast.FunctionDef) and node.name=='sample_farthest_points')]
fps=ast.unparse(tree)+'\n\nsample_farthest_points = sample_farthest_points_naive\n'
(compat/'ops/sample_farthest_points.py').write_text('# BSD reference FPS from PyTorch3D v0.7.4; compiled entry excluded.\n'+fps,encoding='utf-8')
shutil.copyfile('inspection_decisions/pytorch3d_cpu_knn_v1.py',compat/'ops/knn.py')
(compat/'ops/__init__.py').write_text('from .knn import knn_points, knn_gather\nfrom .sample_farthest_points import sample_farthest_points\n',encoding='utf-8')
diff=[]
for name in sorted(set(patches)):
    diff.extend(difflib.unified_diff((original/name).read_text(encoding='utf-8').splitlines(True),(port/name).read_text(encoding='utf-8').splitlines(True),fromfile='original/'+name,tofile='cpu_port/'+name))
(out/'cpu_port.patch').write_text(''.join(diff),encoding='utf-8')
manifest={'scope':'Published identity matching and pairwise pose inference; reconstruction output omitted, no retraining, CPU compatibility port NOT an unmodified full end-to-end reproduction',
    'changes':sorted(set(patches)),'sampling':'Exact official PyTorch3D naive/reference FPS; padding retained',
    'knn':'Pure Torch exact L2/L1 nearest-neighbor port, validated against brute-force. Numerical tie-order may differ from compiled kernels.',
    'icp':'Unchanged official PyTorch3D v0.7.4 points_alignment.py and its Python utilities',
    'files_sha256':{str(p.relative_to(port)):hashlib.sha256(p.read_bytes()).hexdigest() for p in port.rglob('*') if p.is_file()}}
(out/'port_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('PORT READY',len(manifest['files_sha256']))
