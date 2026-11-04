"""Generate versioned runner/scorer without changing the failedV13 sources."""
from pathlib import Path


def main():
    runner=Path('scripts/run_material_surface_correspondence_v13.py').read_text(encoding='utf-8')
    runner=runner.replace('from scripts.prepare_material_surface_charts_v13 import configuration,setup','from scripts.prepare_material_surface_charts_v13 import configuration as original_configuration,setup\n\ndef configuration():\n    cfg=original_configuration();cfg.update(json.loads(Path(\'configs/material_surface_correspondence_v13b.json\').read_text(encoding=\'utf-8\')));return cfg')
    runner=runner.replace('from inspection_decisions.material_surface_correspondence_v13 import ChartPair,ChartTriple,fit_correspondences,PAIRS','from inspection_decisions.material_surface_correspondence_v13 import ChartTriple,fit_correspondences,PAIRS\nfrom inspection_decisions.material_surface_correspondence_v13b import ChartPair')
    runner=runner.replace("'scripts/score_material_surface_correspondence_v13.py'","'scripts/score_material_surface_correspondence_v13b.py'")
    runner=runner.replace("'configs/material_surface_correspondence_v13.json'];sources=", "'configs/material_surface_correspondence_v13.json','configs/material_surface_correspondence_v13b.json','inspection_decisions/material_surface_correspondence_v13b.py','scripts/check_material_surface_correspondence_v13b.py','scripts/prepare_material_surface_correspondence_repair_v13b.py'];sources=")
    scorer=Path('scripts/score_material_surface_correspondence_v13.py').read_text(encoding='utf-8').replace('from scripts.run_material_surface_correspondence_v13 import','from scripts.run_material_surface_correspondence_v13b import').replace('from inspection_decisions.material_surface_correspondence_v13 import ChartPair','from inspection_decisions.material_surface_correspondence_v13b import ChartPair')
    for path,source in [('scripts/run_material_surface_correspondence_v13b.py',runner),('scripts/score_material_surface_correspondence_v13b.py',scorer)]:
        p=Path(path);assert not p.exists(),path;p.write_text(source,encoding='utf-8')
    print('V13B runner/scorer generated; originals unchanged')


if __name__=='__main__':main()
