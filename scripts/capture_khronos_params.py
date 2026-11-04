"""Save parameters through argv-safe rosparam, avoiding upstream shell path quoting."""
import os,pathlib,subprocess
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
shell='source /opt/ros/noetic/setup.bash\nexport ROS_MASTER_URI=http://127.0.0.1:11321\nrosparam dump "$1/results/khronos_reproduction_v1/run_001/supervisor_rosparams.yaml"'
r=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'params',project],capture_output=True)
pathlib.Path('results/khronos_reproduction_v1/run_001/parameter_dump.log').write_bytes(r.stdout+r.stderr)
print('Parameter capture returncode:',r.returncode)
if r.returncode:raise RuntimeError('Parameter capture failed')
