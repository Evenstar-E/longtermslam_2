"""Pixel provenance and old/new evidence preservation for both full controls."""
import json
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_control_v11 import configuration,setup
from scripts.run_textured_material_comparison_v1 import save,sha
from inspection_decisions.textured_material_matching_v1 import world_points

def main():
    cfg=configuration();roots=[Path('results/material_oriented_surface_control_v10'),Path(cfg['output'])];checks=0;pixels=0
    old=json.loads(Path('results/material_surface_patch_registration_v9_run2/completion_integrity.json').read_text(encoding='utf-8'))
    for g in ['source_hashes','evidence_hashes']:
        for p,h in old[g].items():assert sha(p)==h,p;checks+=1
    for root in roots:
        s=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert s['complete'] and not s['private_labels_read'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(root/p)==h;checks+=1
        assert json.loads((root/'scores.json').read_text(encoding='utf-8'))['primary_rows']==384;checks+=1
    for v in cfg['variants']:
        _,frames,_,_,_,_,_,banks,_=setup(cfg,v)
        with np.load(roots[1]/f'{v}_patches.npz') as f:stored=dict(f)
        for e,row in enumerate(banks):
            for i,a in enumerate(row):
                for field in ['patch_xyz','patch_rgb','patch_uv','patch_frame','patch_center_index']:assert np.array_equal(stored[f'{e}_{i}_{field}'],a[field]);checks+=1
                for h,fi in enumerate(a['patch_frame']):
                    f=frames[e][int(fi)];p=a['patch_uv'][h];assert np.array_equal(f['rgb'][p[:,1],p[:,0]]/255.,a['patch_rgb'][h]) and np.max(abs(world_points(f,p)-a['patch_xyz'][h]))<1e-12 and (f['anonymous_slot'][p[:,1],p[:,0]]==i).all();pixels+=len(p);checks+=1
    score192='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';assert sha('results/surface_measure_first_return_v1/scores.json')==score192;checks+=1
    report=Path('research/material_oriented_inverse_results_v11_2026-10-27.md');assert report.is_file();checks+=1
    files=[p for root in roots for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,Path('results/material_oriented_surface_check_v10.json'),Path('results/material_inverse_plane_check_v11.json')]
    sources=['configs/material_oriented_surface_control_v10.json','configs/material_inverse_plane_control_v11.json','inspection_decisions/material_oriented_surface_control_v10.py','inspection_decisions/material_inverse_plane_control_v11.py','scripts/run_material_oriented_surface_control_v10.py','scripts/run_material_inverse_plane_control_v11.py','scripts/score_material_oriented_surface_control_v10.py','scripts/score_material_inverse_plane_control_v11.py','scripts/check_material_oriented_surface_control_v10.py','scripts/check_material_inverse_plane_control_v11.py','scripts/prepare_material_inverse_plane_control_v11.py','scripts/prepare_material_inverse_plane_evaluation_v11.py','scripts/diagnose_material_view_support_v10.py','scripts/diagnose_material_view_support_v11.py','scripts/diagnose_material_stencil_loss_v10.py','scripts/report_material_oriented_inverse_controls_v11.py',__file__]
    save(roots[1]/'completion_integrity.json',dict(complete=True,checks=checks,pixels_verified=pixels,primary_rows=768,TRAIN_scenes=1,new_poses=False,new_training=False,old_evidence_unchanged=True,old_192_score_sha=score192,PNG_opened_and_reviewed=True,figure_review='Both full stages/all4 methods/4appearances, identical0-24axes, exactcounts, no clipping. N=1 TRAIN and fixed oldpool limit explicit.',source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files}));print('V10/V11 COMPLETION',checks,'PIXELS',pixels)

if __name__=='__main__':main()
