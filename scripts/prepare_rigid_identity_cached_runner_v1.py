"""Create isolated equivalent runtime runner, keeping the initial frozen source."""
from pathlib import Path
source=Path('scripts/run_rigid_identity_bounds_v1.py').read_text(encoding='utf-8')
source=source.replace('from inspection_decisions.rigid_identity_bounds_v1 import JointProblem,pose','from inspection_decisions.rigid_identity_bounds_v1 import pose\nfrom inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem as JointProblem')
source=source.replace("Path('results/rigid_identity_bounds_v1')","Path('results/rigid_identity_cached_v1')")
source=source.replace("Path('inspection_decisions/rigid_identity_bounds_v1.py'),cfgpath]","Path('inspection_decisions/rigid_identity_bounds_v1.py'),Path('inspection_decisions/rigid_identity_cached_v1.py'),cfgpath]")
Path('scripts/run_rigid_identity_cached_v1.py').write_text(source,encoding='utf-8')
# Downstream readers must explicitly follow the completed cached run, not the
# interrupted engineering run. Do not pretend its unfinished output is failure0.
for name in ['run_rigid_identity_cut_ablation_v1.py','score_rigid_identity_bounds_v1.py']:
    p=Path('scripts')/name;p.write_text(p.read_text(encoding='utf-8').replace("Path('results/rigid_identity_bounds_v1')","Path('results/rigid_identity_cached_v1')").replace("'results/rigid_identity_bounds_v1/scores.json'","'results/rigid_identity_cached_v1/scores.json'"),encoding='utf-8')
