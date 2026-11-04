"""Prepare a separate native-WSL build workspace; keep logs in shared project."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--prepare",action="store_true");a=p.parse_args()
project=subprocess.check_output(["wsl","-d","Ubuntu-20.04","--exec","pwd"],text=True).strip()
task_home=subprocess.check_output(["wsl","-d","Ubuntu-20.04","--exec","printenv","HOME"],text=True).strip()
ws=task_home+"/uav_khronos_ros1_ws_20261021"
out=Path("results/khronos_reproduction_v1")
log=out/(a.label+".log")
if log.exists():raise FileExistsError(log)
if a.prepare:
    script=r"""
import pathlib,shutil,sys
project=pathlib.Path(sys.argv[1]); ws=pathlib.Path(sys.argv[2])
ws.mkdir(exist_ok=False); (ws/'src').mkdir()
roots=[project/'third_party/Khronos_ROS1']+sorted((project/'third_party/khronos_ros1_deps').iterdir())
for root in roots:
    if not root.is_dir():continue
    dest=ws/'src'/('khronos' if root.name=='Khronos_ROS1' else root.name)
    shutil.copytree(root,dest,ignore=shutil.ignore_patterns('.git'))
    print('copied',root.name,flush=True)
"""
    subprocess.run(["wsl","-d","Ubuntu-20.04","--exec","python3","-c",script,project,ws],check=True,timeout=180)
shell=r"""
set -e
source /opt/ros/noetic/setup.bash
cd "$1"
catkin init
catkin config --extend /opt/ros/noetic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -DCONFIG_UTILS_BUILD_TESTS=OFF -DCONFIG_UTILS_BUILD_DEMOS=OFF -DSPARK_DSG_BUILD_PYTHON=OFF -DSPARK_DSG_BUILD_TESTS=OFF -DSPARK_DSG_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DBUILD_DOC=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TEASER_PYTHON=OFF -DBUILD_TEASER_FPFH=OFF -DKIMERA_VERIFY_GTSAM_CONFIG=OFF -DOPENGV_BUILD_WITH_MARCH_NATIVE=OFF -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin build khronos_ros -j 2 -p 1 --no-status --no-notify --summarize
"""
args=["wsl","-d","Ubuntu-20.04","--exec","bash","-c",shell,"build",ws]
started=time.monotonic()
with log.open("xb") as f:
    proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    while proc.poll() is None:
        elapsed=time.monotonic()-started
        (out/(a.label+"_monitor.json")).write_text(json.dumps({"running":True,"elapsed_s":elapsed,"timeout_s":2400,"workspace":ws}),encoding="utf-8")
        if elapsed>2400:
            proc.terminate();proc.wait(timeout=30);break
        time.sleep(5)
result={"returncode":proc.returncode,"elapsed_s":time.monotonic()-started,"workspace":ws,"log":str(log),"running":False,
        "root_revision":"4e484c4879a76a37e4bf6304b5c58132dc11b279","parallelism":"2 compiler jobs,1 package"}
with (out/(a.label+"_result.json")).open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
print(json.dumps(result))

