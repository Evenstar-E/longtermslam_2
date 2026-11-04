"""Read-only summaries from the completed, no-GT restricted information diagnostic."""
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read, save, sha

def main():
    root=Path('results/translation_surface_information_v2')
    rows=read(root/'rows.json');checks=read(root/'finite_difference.json')
    original=read(root/'manifest.json')
    assert all(sha(p)==v for p,v in original['inputs_unchanged'].items())
    by_scene={}
    for scene in ['scene_a','scene_c']:
        rr=[r for r in rows if r['scene']==scene]
        matrices=np.load(root/f'{scene}_matrices.npz')
        info=[];motion=[]
        for i,r in enumerate(rr):
            C=matrices[f'{i}_complete'][:6,:6]
            P=matrices[f'{i}_profile']
            D=np.diag(1/np.sqrt(np.diag(C)))
            info.append(float(np.linalg.eigvalsh(D@P@D).min()))
            g=matrices[f'{i}_gradient'];step=.001*g/np.linalg.norm(g)
            motion.append(float(np.linalg.norm(step[3:6]-step[:3])*1000))
        by_scene[scene]={'entries':len(rr),'negative_curvature':sum(r['negative_curvature_relative_1e-8'] for r in rr),
                         'gradient_gain_range':[min(r['gradient_step_loglikelihood_delta'] for r in rr),max(r['gradient_step_loglikelihood_delta'] for r in rr)],
                         'complete_profile_min_relative_information_range':[min(info),max(info)],
                         'interperiod_anchor_translation_perturbation_mm_range':[min(motion),max(motion)]}
    out={'by_scene':by_scene,'checks':len(checks),'max_gradient_relative_error':max(c['gradient_relative_error'] for c in checks),
         'max_curvature_relative_error':max(c['curvature_relative_error'] for c in checks),
         'note':'Perturbations are not error reductions; curvature is not calibrated uncertainty; 43 entries include background/new/nonpersistent cases, distinct from 37 persistent evaluation assets.'}
    save(root/'details.json',out)
    save(root/'evidence_index.json',{'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evidence_index.json'}})
    print(out)

if __name__=='__main__':main()
