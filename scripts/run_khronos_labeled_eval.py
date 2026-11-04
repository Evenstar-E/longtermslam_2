"""Evaluate the preserved apartment map with official code, in a fresh experiment directory."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
base=Path('results/labeled_temporal_v1');out=base/a.name;out.mkdir(exist_ok=False)
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
gt=project+'/data/labeled_temporal_v1/ground_truth/apartment/'
source=Path('third_party/Khronos_ROS1/khronos_eval/config/pipeline/apartment.yaml')
cfg=yaml.safe_load(source.read_text())
cfg['ground_truth_dsg_file']=gt+'gt_dsg_consolidated.json'
cfg['ground_truth_background_file']=gt+'gt_background.ply'
cfg['gt_changes_file']=gt+'gt_changes.csv';cfg['evaluation']['object_evaluation']['changes_file']=gt+'gt_changes.csv'
config=out/'official_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
old=Path('results/khronos_reproduction_v1/run_001/algorithm')
os.link(old/'final.4dmap',out/'final.4dmap')
shutil.copy2(old/'experiment_log.txt',out/'experiment_log.txt')
record={'scene':'apartment','protocol_sha256':hashlib.sha256(Path('configs/labeled_temporal_v1.json').read_bytes()).hexdigest(),
 'official_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'adapted_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
 'adaptation':'Only ground-truth paths; unchanged official metric and algorithm code; fresh experiment directory with read-only map hardlink.',
 'source_map':'results/khronos_reproduction_v1/run_001/algorithm/final.4dmap',
 'metric_caveats':['Repeated identical map/query rows in official evaluator must be deduplicated in interpretation.',
    'GT change epoch not yet verified against bag clock. Raw official outputs are diagnostic.',
    'Default change metrics exclude unassociated missed/hallucinated objects; report inclusive counts separately.']}
(out/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
shell='''set -e
source /opt/ros/noetic/setup.bash
source /home/tong/uav_khronos_ros1_ws_20261021/devel/setup.bash
exec /usr/bin/time -v /home/tong/uav_khronos_ros1_ws_20261021/devel/lib/khronos_eval/exp_pipeline "$1/official_config.yaml" "$1" false true false
'''
started=time.monotonic()
with (out/'evaluation.log').open('xb') as log:
    proc=subprocess.Popen(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'eval',project+'/'+out.as_posix()],stdout=log,stderr=subprocess.STDOUT)
    while proc.poll() is None:
        elapsed=time.monotonic()-started
        (out/'monitor.json').write_text(json.dumps({'running':True,'elapsed_s':elapsed,'timeout_s':1200}),encoding='utf-8')
        if elapsed>1200:
            proc.terminate();proc.wait(timeout=30);break
        time.sleep(5)
    files={n:(out/'results'/n).stat().st_size if (out/'results'/n).exists() else 0 for n in ['static_objects.csv','dynamic_objects.csv','background_mesh.csv','map_timestamps.txt']}
record={'returncode':proc.returncode,'elapsed_s':time.monotonic()-started,'output_sizes':files,
        'complete':proc.returncode==0 and all(files.values()) and 'All evaluation runs completed.' in (out/'evaluation.log').read_text(errors='replace',encoding='utf-8')}
(out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(out/'monitor.json').write_text(json.dumps({'running':False,**record}),encoding='utf-8')
print(json.dumps(record),flush=True)
