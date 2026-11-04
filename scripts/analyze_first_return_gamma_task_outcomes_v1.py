"""Post-seal task strata; never fits or modifies frozen predictions."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.completion_catalog_scene_v1 import ray_box,transform
from scripts.first_return_surface_scene_v1 import geometry_for


def main():
    root=Path('results/first_return_gamma_material_v5');assert read(root/'audit.json')['passed'] and (root/'evaluation_sources.json').exists();cfg=read(root/'config.json');scores=read(root/'scores.json');norms={};details=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=Path(cfg['base_root'])/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        G=geometry_for(run['seed'],24,run['geometry'])
        for i in range(24):
            ray=inp['historical_rays'][i];o=np.array(ray['origin']);r=np.array(ray['ray']);z=min(float(ray_box(o,r,lo,hi)) for lo,hi in G[i]);material=o+z*r;p=[transform(material[None],T)[0] for T in truth['poses'][:,i]];d=float(np.linalg.norm(p[1]-p[0]));norms[tag,i]=d;details.append({'scene':tag,'asset':i,'true_material_displacement_m':d,'stratum':'gt_displacement_gt_5cm' if d>.05 else 'gt_displacement_le_5cm'})
    groups=[]
    for family,method in [('hazard',m) for m in cfg['methods']]+[('catalogue_plane','previous_typed_ray'),('Gibbs','previous_joint_ray')]:
        for moving in [False,True]:
            rr=[r for r in scores if (r['family'],r['method'],r['stage'])==(family,method,'final') and (norms[r['scene'],r['asset']]>.05)==moving];groups.append({'family':family,'method':method,'true_displacement_gt_5cm':moving,'n':len(rr),'success':sum(r['success'] for r in rr),'both_ids':sum(r['both_ids'] for r in rr),'wrong_ID':sum(not r['both_ids'] for r in rr),'wrong_quantification_given_correct_ID':sum(r['both_ids'] and not r['success'] for r in rr)})
    paired=[]
    for moving in [False,True]:
        a=[r for r in scores if (r['family'],r['method'],r['stage'])==('hazard','gamma_joint','final') and (norms[r['scene'],r['asset']]>.05)==moving];lookup={(r['scene'],r['asset']):r for r in scores if (r['family'],r['method'],r['stage'])==('hazard','gamma_independent','final')};b=[lookup[r['scene'],r['asset']] for r in a];paired.append({'true_displacement_gt_5cm':moving,'n':len(a),'gained':sum(x['success'] and not y['success'] for x,y in zip(a,b)),'lost':sum(y['success'] and not x['success'] for x,y in zip(a,b))})
    save(root/'task_outcome_strata.json',{'script_sha256':sha(Path(__file__)),'postseal_diagnosis_only':True,'groups':groups,'joint_vs_independent':paired,'motion_values':details});print('TASK_STRATA',groups,'PAIRED',paired,flush=True)


if __name__=='__main__':main()
