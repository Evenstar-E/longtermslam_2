"""Launch the isolated ROS1 example with native ROS Python, not the Windows venv."""
import argparse,subprocess
p=argparse.ArgumentParser();p.add_argument("--run-name",required=True);a=p.parse_args()
project=subprocess.check_output(["wsl","-d","Ubuntu-20.04","--exec","pwd"],text=True).strip()
task_home=subprocess.check_output(["wsl","-d","Ubuntu-20.04","--exec","printenv","HOME"],text=True).strip()
ws=task_home+"/uav_khronos_ros1_ws_20261021"
shell='source /opt/ros/noetic/setup.bash\nsource "$1/devel/setup.bash"\nexec /usr/bin/python3 "$2/scripts/run_khronos_example_wsl.py" --workspace "$1" --project "$2" --run-name "$3"'
subprocess.run(["wsl","-d","Ubuntu-20.04","--exec","bash","-c",shell,"run",ws,project,a.run_name],check=True)

