"""Same three scene_a archive arms under a common, isolated progress repair."""
import argparse
import json
from pathlib import Path
from scripts.run_rescan_third_period_v1 import run,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['initial_archive','inherited_unfused','inherited_protected']);args=ap.parse_args()
    root=Path('results/rescan_progress_port_v1');cfg=json.loads(Path('configs/rescan_third_period_v1.json').read_text(encoding='utf-8'))
    cfg['binaries']='third_party/Rescan_progress_port_v1/bin'
    cfg['modes']=[args.mode]
    cfg['scope']='scene_a common progress-repaired engine for ALL three archive arms. Original runs retained. No evaluation labels used; same parameters, no geometric/objective changes. Not full Poisson reconstruction.'
    cfg['repair_source_sha']=sha('third_party/Rescan_progress_port_v1/lib/rs/rs_pointcloud.h')
    cfg['runner_source_sha']=sha(__file__)
    if args.mode=='inherited_protected':cfg['prior_roots']['scene_a']='data/rescan_protected_archive_v1/scene_a'
    config=root/f'configuration_{args.mode}.json';assert not config.exists();save(config,cfg)
    run('scene_a',args.mode,cfg,config,Path('results/rescan_third_period_progress_v1'))


if __name__=='__main__':main()
