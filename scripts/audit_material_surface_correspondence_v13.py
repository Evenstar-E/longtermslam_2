"""Old evidence, actual measured chart colors/masks, final V13 seals."""
import json
from pathlib import Path
import numpy as np
from scripts.run_material_surface_correspondence_v13b import configuration,setup
from scripts.cache_material_surface_geometry_v9 import unpack
from inspection_decisions.material_inverse_plane_control_v11 import project
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);parent=Path(cfg['previous_root']);old=json.loads((parent/'completion_integrity.json').read_text(encoding='utf-8'));checks=0
    for group in ['source_hashes','evidence_hashes']:
        for p,h in old[group].items():assert sha(p)==h,p;checks+=1
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    chartroot=Path(cfg['chart_cache']);c=json.loads((chartroot/'chart_seal.json').read_text(encoding='utf-8'))
    for p,h in c['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in c['files'].items():assert sha(chartroot/p)==h,p;checks+=1
    queries=0;valid=0
    for v in cfg['variants']:
        _,frames,_,_,_,_,_,_,_=setup(cfg,v);charts=unpack(chartroot/f'{v}_charts.npz')
        for e,row in enumerate(charts):
            for slot,a in enumerate(row):
                for fi in np.unique(a['frame']):
                    ix=a['frame']==fi;q=project(a['ring_xyz'][ix],frames[e][int(fi)],slot);mask=q['surface']&(abs(q['gap'])<=cfg['chart_self_tolerance_m'])&(np.sum(q['normal']*a['basis'][ix,None,None,:,2],axis=-1)>.7)
                    assert np.array_equal(mask,a['mask'][ix]);assert np.max(abs(q['rgb'][mask]-a['rgb'][ix][mask]),initial=0)<1e-12;queries+=mask.size;valid+=int(mask.sum());checks+=2
    failed=Path('results/material_surface_correspondence_v13');frozen=json.loads((failed/'frozen_protocol.json').read_text(encoding='utf-8'))
    for p,h in frozen['sources'].items():assert sha(p)==h,p;checks+=1
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));assert scores['primary_rows']==768 and scores['max_selected_objective_error']<1e-8;checks+=1
    old192='results/surface_measure_first_return_v1/scores.json';assert sha(old192)=='196d1f678b8496c512756942ecfd7ed028f6da86d393ffc43fc990e311955dd4';checks+=1
    report=Path('research/material_surface_correspondence_results_v13_2026-10-27.md');assert report.is_file() and (root/'task_comparison.png').is_file();checks+=1
    sources=list(seal['sources'])+['scripts/score_material_surface_correspondence_v13c.py','scripts/prepare_material_surface_scoring_repair_v13c.py','scripts/diagnose_material_chart_coverage_v13.py','scripts/diagnose_material_sampling_loss_v13.py','scripts/analyze_material_surface_mainline_v13.py','scripts/check_material_chart_center_shift_v13.py','scripts/report_material_surface_correspondence_v13.py',__file__]
    files=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json']+[report,Path('research/material_surface_correspondence_model_v13_2026-10-27.md'),Path('results/material_surface_correspondence_checks_v13.json'),Path('results/material_surface_correspondence_periodic_checks_v13b.json'),Path('results/material_chart_center_shift_counterexample_v13.json'),failed/'failure_note.json',chartroot/'chart_seal.json']
    save(root/'completion_integrity.json',dict(complete=True,checks=checks,primary_rows=768,actual_texture_queries_verified=queries,valid_texture_queries_verified=valid,old_evidence_unchanged=True,old_192_sha=sha(old192),new_training=False,TRAIN_scenes=1,source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files},figure_review='PNG opened and reviewed separately, allmethods/appearances/same0-24scale/counts; PDFexported.'))
    print('V13 COMPLETION',checks,'QUERIES',queries,'VALID',valid)


if __name__=='__main__':main()
