"""Isolated mass-bound version; score, data, pose leaves and feasible UB unchanged."""
import ast
from pathlib import Path
source=Path('inspection_decisions/rigid_identity_cuts_v1.py').read_text(encoding='utf-8');tree=ast.parse(source)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='build_conflicts')
lines=source.splitlines(True)
lines[node.lineno-1:node.end_lineno]=['    def build_conflicts(self):\n','        from inspection_decisions.rigid_identity_mass_geometry_v1 import mass_conflicts\n','        self.conflicts=mass_conflicts(self.H,self.C,self.radius,self.cfg["distance_scale_m"],self.mass)\n','        return self.conflicts\n']
source=''.join(lines).replace('class RigidityCutsProblem','class MassRigidityCutsProblem').replace('for i,j,l,_,_ in self.conflicts:','for i,j,l,_,_,fj,fl in self.conflicts:').replace("1/(self.cfg['geometry_weight']*self.mass[j])","1/(self.cfg['geometry_weight']*self.mass[j]*fj)").replace("1/(self.cfg['geometry_weight']*self.mass[l])","1/(self.cfg['geometry_weight']*self.mass[l]*fl)")
Path('inspection_decisions/rigid_identity_mass_cuts_v1.py').write_text(source,encoding='utf-8')
runner=Path('scripts/run_rigid_identity_cut_ablation_v1.py').read_text(encoding='utf-8')
runner=runner.replace('from inspection_decisions.rigid_identity_cuts_v1 import RigidityCutsProblem','from inspection_decisions.rigid_identity_mass_cuts_v1 import MassRigidityCutsProblem as RigidityCutsProblem')
runner=runner.replace('results/rigid_identity_cuts_v1_completed','results/rigid_identity_mass_cuts_v1')
runner=runner.replace("Path('inspection_decisions/rigid_identity_cuts_v1.py')","Path('inspection_decisions/rigid_identity_mass_cuts_v1.py'),Path('inspection_decisions/rigid_identity_mass_geometry_v1.py')")
runner=runner.replace("'historical_diameter_upper_m':d} for i,j,k,s,d in p.conflicts","'historical_diameter_upper_m':d,'fraction_a':fa,'fraction_b':fb} for i,j,k,s,d,fa,fb in p.conflicts")
Path('scripts/run_rigid_identity_mass_ablation_v1.py').write_text(runner,encoding='utf-8')
