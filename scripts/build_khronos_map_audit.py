"""Compile a read-only saved-map inspector against the reproduced libraries."""
import pathlib,subprocess
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
shell='''set -e
source /opt/ros/noetic/setup.bash
source /home/tong/uav_khronos_ros1_ws_20261021/devel/setup.bash
cmake -S "$1/tools/khronos_map_audit" -B /home/tong/uav_khronos_ros1_ws_20261021/map_audit_build -DCMAKE_BUILD_TYPE=Release
cmake --build /home/tong/uav_khronos_ros1_ws_20261021/map_audit_build -j 1
'''
out=pathlib.Path('results/khronos_reproduction_v1/map_audit_build.log')
with out.open('xb') as f:
    p=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'audit',project],stdout=f,stderr=subprocess.STDOUT,timeout=180)
print('Inspector build returncode:',p.returncode)
if p.returncode:raise RuntimeError('Map inspector build failed; see log')
