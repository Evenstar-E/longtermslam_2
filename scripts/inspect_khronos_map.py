"""Load the completed official map through its public C++ API; keep a log."""
import pathlib,subprocess
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
shell='''set -e
source /opt/ros/noetic/setup.bash
source /home/tong/uav_khronos_ros1_ws_20261021/devel/setup.bash
exec /home/tong/uav_khronos_ros1_ws_20261021/map_audit_build/khronos_map_audit "$1/results/khronos_reproduction_v1/run_001/algorithm/final.4dmap" "$1/results/khronos_reproduction_v1/run_001/map_content.json"
'''
with pathlib.Path('results/khronos_reproduction_v1/run_001/map_inspection.log').open('xb') as f:
    p=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'inspect',project],stdout=f,stderr=subprocess.STDOUT,timeout=180)
print('Map inspection returncode:',p.returncode)
if p.returncode:raise RuntimeError('Map inspection failed')
