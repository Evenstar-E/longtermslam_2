"""Build and run an explicit-query driver; preserve official mapper and metric code."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
out=Path('results/labeled_temporal_v1')/a.name;out.mkdir(exist_ok=False)
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
info=json.loads(Path('results/khronos_reproduction_v1/run_001/bag_info.json').read_text())
offset=round(info['start_s']*1e9)
shell='''set -e
source /opt/ros/noetic/setup.bash
source /home/tong/uav_khronos_ros1_ws_20261021/devel/setup.bash
cmake -S "$1/tools/temporal_query_audit" -B /home/tong/uav_khronos_ros1_ws_20261021/temporal_query_audit_build -DCMAKE_BUILD_TYPE=Release
cmake --build /home/tong/uav_khronos_ros1_ws_20261021/temporal_query_audit_build -j 1
exec /usr/bin/time -v /home/tong/uav_khronos_ros1_ws_20261021/temporal_query_audit_build/temporal_query_audit "$1/results/khronos_reproduction_v1/run_001/algorithm/final.4dmap" "$1/data/labeled_temporal_v1/ground_truth/apartment/gt_dsg_consolidated.json" "$2" "$3"
'''
start=time.monotonic()
with (out/'execution.log').open('xb') as log:
    p=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'query',project,project+'/'+out.as_posix(),str(offset)],stdout=log,stderr=subprocess.STDOUT,timeout=900)
record={'returncode':p.returncode,'elapsed_s':time.monotonic()-start,'static_shift_sensitivity_ns':offset,
 'config_sha256':hashlib.sha256(Path('configs/temporal_query_audit_v1.json').read_bytes()).hexdigest(),
 'driver_sha256':hashlib.sha256(Path('tools/temporal_query_audit/main.cpp').read_bytes()).hexdigest(),
 'scope':'Evaluation-driver adaptation/temporal interval ablation only; not an independent baseline or verified label-epoch correction'}
(out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
