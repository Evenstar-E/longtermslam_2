"""Read final map IDs/geometry metadata through the public native C++ API."""
import json,subprocess,time
from pathlib import Path
out=Path('results/event_evidence_v1');logpath=out/'prediction_export.log'
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
shell='''set -e
source /opt/ros/noetic/setup.bash
source /home/tong/uav_khronos_ros1_ws_20261021/devel/setup.bash
cmake -S "$1/tools/event_prediction_export" -B /home/tong/uav_khronos_ros1_ws_20261021/event_prediction_export_build -DCMAKE_BUILD_TYPE=Release
cmake --build /home/tong/uav_khronos_ros1_ws_20261021/event_prediction_export_build -j 1
exec /home/tong/uav_khronos_ros1_ws_20261021/event_prediction_export_build/event_prediction_export "$1/results/khronos_reproduction_v1/run_001/algorithm/final.4dmap" "$1/results/event_evidence_v1/prediction_nodes.json"
'''
start=time.monotonic()
with logpath.open('xb') as f:
    r=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'ids',project],stdout=f,stderr=subprocess.STDOUT,timeout=180)
(out/'prediction_export_result.json').write_text(json.dumps({'returncode':r.returncode,'elapsed_s':time.monotonic()-start},indent=2),encoding='utf-8')
print('Prediction export returncode',r.returncode,flush=True)
