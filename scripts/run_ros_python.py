"""Run a project script in native ROS Python without starting a ROS master."""
import argparse,subprocess
p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('args',nargs='*');a=p.parse_args()
project=subprocess.check_output(['wsl','-d','Ubuntu-20.04','--exec','pwd'],encoding='utf-8').strip()
shell='source /opt/ros/noetic/setup.bash\nexport PYTHONPATH="$1:$PYTHONPATH"\nexec /usr/bin/python3 "$1/$2" "${@:3}"'
subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell,'offline',project,a.script,*a.args],check=True)
