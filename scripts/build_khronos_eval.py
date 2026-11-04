"""Build the pinned official evaluation package using the existing native workspace."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--attempt',default='build_eval');a=p.parse_args()
out=pathlib.Path('results/labeled_temporal_v1');out.mkdir(exist_ok=True)
if (out/(a.attempt+'.log')).exists():raise FileExistsError(a.attempt)
shell='''set -e
source /opt/ros/noetic/setup.bash
cd /home/tong/uav_khronos_ros1_ws_20261021
catkin build khronos_eval -j 2 -p 1 --no-status --no-notify --summarize
'''
start=time.monotonic()
with (out/(a.attempt+'.log')).open('xb') as f:
    p=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
record={'returncode':p.returncode,'elapsed_s':time.monotonic()-start,'root_revision':'4e484c4879a76a37e4bf6304b5c58132dc11b279','algorithm_source_modified':False}
(out/(a.attempt+'_result.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
