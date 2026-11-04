"""Run under sourced ROS1 Python in WSL. Supervise only our dedicated ROS master."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--project",type=Path,required=True);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--run-name",required=True)
a=p.parse_args();project=a.project
out=project/"results/khronos_reproduction_v1"/a.run_name
out.mkdir(exist_ok=False)
bag_path=project/"data/khronos_official_v1/tesse_cd_apartment.bag"
acq=json.loads((bag_path.parent/"acquisition.json").read_text())
if not acq["complete"] or bag_path.stat().st_size!=acq["bytes"]:raise RuntimeError("Official bag incomplete")
import rosbag
with rosbag.Bag(str(bag_path)) as bag:
    details=bag.get_type_and_topic_info()
    info={"start_s":bag.get_start_time(),"end_s":bag.get_end_time(),"messages":bag.get_message_count(),
       "topics":{k:{"type":v.msg_type,"messages":v.message_count,"frequency":v.frequency} for k,v in details.topics.items()}}
duration=info["end_s"]-info["start_s"]
(out/"bag_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
required=["/tesse/left_cam/rgb/image_raw","/tesse/depth_cam/mono/image_raw","/tesse/seg_cam/converted/image_raw"]
if any(k not in info["topics"] for k in required):raise RuntimeError("Official input topics do not match launch")
import socket
port=11321
while port<11331:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",port))!=0:break
    port+=1
if port>=11331:raise RuntimeError("No free dedicated ROS port")
uri="http://127.0.0.1:"+str(port)
os.environ["ROS_MASTER_URI"]=uri;os.environ["ROS_IP"]="127.0.0.1"
os.environ["ROS_LOG_DIR"]=str(out/"ros_logs")
env=os.environ.copy()
args=["roslaunch","--port",str(port),str(project/"configs/khronos_headless_apartment_v1.launch"),
      "bag_dir:="+str(bag_path.parent),"output_dir:="+str(out/"algorithm"),
      "start_visualizer:=false","paused:=true","play_rate:=0.5","sim_time_required:=true"]
started=time.monotonic();launch=None;result={"complete":False,"ROS_MASTER_URI":uri,"argv":args,
    "privileged_inputs":"Official simulated example with GT poses and semantic labels; not visual localization benchmark.",
    "playback_rate":.5,"bag_duration_s":duration,"algorithm_config_modified":False}
log_path=out/"launch.log"
try:
    log=log_path.open("wb")
    launch=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    import rospy,rosnode,rosservice
    from std_srvs.srv import SetBool
    from sensor_msgs.msg import CameraInfo
    # Pipeline construction waits for CameraInfo before advertising its finish service.
    # Replay only the bag's immutable calibration while its image stream stays paused.
    for _ in range(60):
        try:
            rosservice.get_service_list();break
        except Exception:time.sleep(1)
    rospy.init_node("khronos_reproduction_supervisor",anonymous=True,disable_signals=True)
    with rosbag.Bag(str(bag_path)) as bag:
        camera_info=next(bag.read_messages(topics=["/tesse/left_cam/camera_info"]))[1]
    calibration_pub=rospy.Publisher("/tesse/left_cam/camera_info",CameraInfo,queue_size=1,latch=True)
    calibration_pub.publish(camera_info)
    result["startup_calibration"]="First original CameraInfo, latched while bag paused; no image/pose injected."
    # Master may not be ready at process start.
    ready=False
    for _ in range(120):
        if launch.poll() is not None:raise RuntimeError("roslaunch exited during startup")
        try:
            services=rosservice.get_service_list()
            if "/khronos_node/experiment/finish_mapping_and_save" in services:
                ready=True;break
        except Exception:pass
        time.sleep(1)
    if not ready:raise RuntimeError("Mapping service startup timeout")
    dump=subprocess.run(["rosparam","dump",str(out/"supervisor_rosparams.yaml")],capture_output=True,env=env,timeout=30)
    (out/"parameter_dump.log").write_bytes(dump.stdout+dump.stderr)
    result["supervisor_parameter_dump_returncode"]=dump.returncode
    pause="/dataset/pause_playback"
    rospy.wait_for_service(pause,timeout=30)
    answer=rospy.ServiceProxy(pause,SetBool)(False)
    result["unpause_response"]=str(answer)
    playback_start=time.monotonic()
    deadline=playback_start+min(2400,duration/.5+300)
    last_report=0;observed_dataset=False;last_clock=[None]
    from rosgraph_msgs.msg import Clock
    def clock_callback(msg):last_clock[0]=msg.clock.to_sec()
    clock_sub=rospy.Subscriber("/clock",Clock,clock_callback,queue_size=1)
    while True:
        if launch.poll() is not None:raise RuntimeError("roslaunch terminated during playback")
        now=time.monotonic()
        if now>deadline:raise TimeoutError("Hard playback timeout")
        try:names=rosnode.get_node_names()
        except Exception:
            time.sleep(2);continue
        if "/dataset" in names:observed_dataset=True
        if observed_dataset and now-playback_start>5 and "/dataset" not in names:
            break
        if now-last_report>15:
            monitor={"running":True,"elapsed_s":now-started,"phase":"playing","nodes":names,
                     "hard_timeout_s":min(2400,duration/.5+300)}
            (out/"monitor.json").write_text(json.dumps(monitor,indent=2),encoding="utf-8")
            print(json.dumps(monitor),flush=True);last_report=now
        time.sleep(2)
    result["playback_wall_s"]=time.monotonic()-playback_start
    result["dataset_node_exited"]=True
    result["last_clock_s"]=last_clock[0]
    result["reached_bag_end"]=last_clock[0] is not None and last_clock[0]>=info["end_s"]-.2
    if not result["reached_bag_end"]:raise RuntimeError("Dataset disappeared before recorded bag end")
    save=subprocess.run(["rosservice","call","/khronos_node/experiment/finish_mapping_and_save","{}"],
            capture_output=True,text=True,env=env,timeout=300)
    (out/"finish_service.log").write_text(save.stdout+save.stderr,encoding="utf-8")
    result["finish_service_returncode"]=save.returncode
    if save.returncode:raise RuntimeError("Finish/save service failed")
    map_file=out/"algorithm/final.4dmap"
    exp_log=out/"algorithm/experiment_log.txt"
    result["map_bytes"]=map_file.stat().st_size if map_file.exists() else 0
    result["clean_flag"]=exp_log.exists() and "Experiment Finished Cleanly" in exp_log.read_text(errors="replace")
    result["complete"]=result["map_bytes"]>0 and result["clean_flag"]
    if not result["complete"]:raise RuntimeError("Missing map or clean finish flag")
except Exception as e:
    result["error"]=repr(e)
finally:
    if launch is not None:
        if launch.poll() is None:
            os.killpg(launch.pid,signal.SIGINT)
            try:launch.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid,signal.SIGTERM)
                try:launch.wait(timeout=10)
                except subprocess.TimeoutExpired:result["cleanup_timeout"]=True
        result["launch_returncode"]=launch.poll()
        log.close()
    result["elapsed_s"]=time.monotonic()-started
    result["map_sha256"]=None
    map_file=out/"algorithm/final.4dmap"
    if map_file.exists():
        h=hashlib.sha256()
        with map_file.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
        result["map_sha256"]=h.hexdigest()
    (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"monitor.json").write_text(json.dumps({"running":False,"elapsed_s":result["elapsed_s"],"complete":result["complete"]}),encoding="utf-8")
    print(json.dumps(result),flush=True)

