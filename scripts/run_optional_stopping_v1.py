"""Fit cross-fitted continuation on old TRAIN only; freeze then fresh RGB-D."""
import argparse
from pathlib import Path
import numpy as np
from inspection_decisions.graph_dataset import digest
from inspection_decisions.optional_stopping_v1 import train_banks,first_stop,NEW
from scripts.run_evidence_value_pilot_v1 import read,save
from scripts.run_report_risk_stopping_v1 import load_paths,aggregate,collect

CFG=Path('configs/optional_stopping_v1.json')


def sources():
    paths=[CFG,Path(__file__),Path('research/optional_stopping_contract_v1_2026-10-23.md'),
           Path('scripts/run_report_risk_stopping_v1.py'),Path('scripts/run_component_set_planner_v1.py'),
           Path('scripts/run_evidence_value_pilot_v1.py'),Path('scripts/partial_surface_scene.py'),
           Path('configs/evidence_graph_dataset_v1.json'),Path('configs/comparative_evidence_interface_v1.json'),
           Path('results/component_set_planner_v1/models.json'),Path('results/report_risk_stopping_v1/models.json'),
           Path('tests/test_optional_stopping_v1.py')]
    paths+=list(Path('inspection_decisions').glob('*.py'))+list(Path('comparative_evidence').glob('*.py'))
    return {str(p):digest(p) for p in paths}


def select_setting(paths,method,bundle,cfg):
    baseline=aggregate([p['labels_data']['metrics'][-1] for p in paths],[5]*len(paths));trials=[]
    for setting in cfg['thresholds']:
        stages=[first_stop(iter(p['observed']['states']),method,bundle,setting,cfg['price'])[0] for p in paths]
        summary=aggregate([p['labels_data']['metrics'][t] for p,t in zip(paths,stages)],stages)
        feasible=(summary['fp']<=baseline['fp'] and summary['bad_motion_reports']<=baseline['bad_motion_reports']
                  and summary['qualified_tp']>=baseline['qualified_tp'] and summary['task_loss']<=baseline['task_loss']+1e-9)
        trials.append(dict(setting=setting,summary=summary,feasible=feasible,stages=stages))
    eligible=[r for r in trials if r['feasible']];assert eligible
    best=min(eligible,key=lambda r:(r['summary']['mean_acquisitions'],r['summary']['task_loss'],r['setting']))
    return best['setting'],dict(method=method,baseline=baseline,trials=trials,selected=best)


def train(cfg):
    out=Path(cfg['output']);old=Path(cfg['training_paths'])
    if out.exists():raise FileExistsError(out)
    oldfreeze=read(old/'frozen_models.json')
    for p,sha in oldfreeze['sources'].items():assert digest(p)==sha,p
    assert digest(cfg['legacy_models'])==oldfreeze['models_sha256']
    for p,sha in oldfreeze['training_manifests'].items():assert digest(p)==sha,p
    out.mkdir(parents=True);save(out/'started_training.json',dict(sources=sources(),config=cfg))
    paths=load_paths(old,'training');assert len(paths)==64
    train=[p for p in paths if p['split']=='train'];cal=[p for p in paths if p['split']=='calibration' and p['policy']=='rank']
    assert {p['scene'] for p in train}==set(range(16)) and {p['scene'] for p in cal}==set(range(16,32))
    legacy=read(cfg['legacy_models']);heads,audit=train_banks(train,legacy,cfg['alpha'],cfg['price'],cfg['folds'])
    bundle=dict(heads=heads,legacy=legacy,settings=dict(legacy['settings']));search=[]
    for method in NEW:
        if method=='optional_plain_temporal':bundle['settings'][method]=None;continue
        setting,detail=select_setting(cal,method,bundle,cfg);bundle['settings'][method]=setting;search.append(detail)
        print(method,'tau',setting,'cal',detail['selected']['summary'],flush=True)
    save(out/'models.json',bundle);save(out/'training_targets.json',audit);save(out/'calibration_search.json',search)
    frozen=dict(sources=sources(),models_sha256=digest(out/'models.json'),targets_sha256=digest(out/'training_targets.json'),
        search_sha256=digest(out/'calibration_search.json'),training_manifests=oldfreeze['training_manifests'],
        train_families=list(range(16)),calibration_families=list(range(16,32)),
        current_training_backend_fits=0,reused_training_calibration_backend_fits=384,evaluation_generated=False)
    save(out/'frozen_models.json',frozen);print('FROZEN before fresh scene generation',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate'])
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();cfg=read(CFG)
    if args.stage=='train':train(cfg)
    else:
        assert 0<=args.shard<args.shards
        collect(cfg,'evaluation',args.shard,args.shards)


if __name__=='__main__':main()
