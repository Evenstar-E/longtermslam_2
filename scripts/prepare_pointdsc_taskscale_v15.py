"""Pre-GT fixed task-scale control; no sweep or weight modification."""
from pathlib import Path


def main():
    # Must be specified before any new V15 task score is available.
    for root in ['results/material_surface_distribution_matching_v15','results/pointdsc_material_reference_v15']:assert not (Path(root)/'scores.json').exists()
    source=Path('scripts/run_pointdsc_material_reference_v15.py').read_text(encoding='utf-8');source=source.replace('from inspection_decisions.pointdsc_reference_v1 import load_reference,estimate','from inspection_decisions.pointdsc_reference_v1 import load_reference as official_load_reference,estimate')
    source=source.replace("METHODS=['pointdsc_top1']", """METHODS=['pointdsc_task05']

def load_reference():
    model,cfg=official_load_reference();cfg=dict(cfg)
    model.inlier_threshold=.05;model.nms_radius=.05
    cfg['inlier_threshold']=.05;cfg['nms_radius']=.05
    return model,cfg
""")
    source=source.replace("c['output']='results/pointdsc_material_reference_v15'", "c['output']='results/pointdsc_taskscale_reference_v15'")
    source=source.replace('Unchanged official PointDSC3DMatch weights+default thresholds','Unchanged official PointDSC3DMatch weights; fixed0.05m task inlier/NMS thresholds specified before any V15GT score; sigma_spat/pretrained weights unchanged')
    scorer=Path('scripts/score_pointdsc_material_reference_v15.py').read_text(encoding='utf-8').replace('from scripts.run_pointdsc_material_reference_v15 import','from scripts.run_pointdsc_taskscale_reference_v15 import').replace('official-demo top1; bidirectional0.1m inlier-count','official-demo top1; preregistered0.05m inlier/NMS and bidirectional0.05m inlier-count')
    for path,code in [('scripts/run_pointdsc_taskscale_reference_v15.py',source),('scripts/score_pointdsc_taskscale_reference_v15.py',scorer)]:
        p=Path(path);assert not p.exists();compile(code,path,'exec');p.write_text(code,encoding='utf-8')
    print('Fixed task-scale control generated BEFORE score access')


if __name__=='__main__':main()
