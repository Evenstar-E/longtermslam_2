"""Bounded environment preparation and public official dataset listing."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("action",choices=["apt","drive"]);a=p.parse_args()
out=Path("results/khronos_reproduction_v1")
if a.action=="apt":
    args=["wsl","-d","Ubuntu-20.04","-u","root","--","env","DEBIAN_FRONTEND=noninteractive",
          "apt-get","-o","Acquire::Retries=0","-o","Acquire::http::Timeout=20","-y",
          "--no-install-recommends","install","python3-catkin-tools","python3-vcstool",
          "ros-noetic-gtsam","libgoogle-glog-dev","nlohmann-json3-dev"]
    with (out/"apt_install.log").open("xb") as f:
        proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        start=time.monotonic()
        try:rc=proc.wait(timeout=480)
        except subprocess.TimeoutExpired:
            proc.terminate();rc=proc.wait()
        result={"argv":args,"returncode":rc,"elapsed_s":time.monotonic()-start,
                "planned_change":"9 new packages,0 upgrades,0 removals per simulation","timeout_s":480}
    (out/"apt_install.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
else:
    import gdown
    start=time.monotonic()
    try:
        rows=gdown.download_folder(id="1VQf-Q8nAuCAsq9wplB07oIpAiWpVVR7U",
                  output=str(out/"official_data"),skip_download=True,use_cookies=False,timeout=25,retries=0)
        result={"status":"listed","rows":[dict(x._asdict()) for x in rows],"elapsed_s":time.monotonic()-start}
    except Exception as e:result={"status":"failed","error":repr(e),"elapsed_s":time.monotonic()-start}
    with (out/"official_drive_index_v2.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,default=str)
    print(json.dumps(result,ensure_ascii=False,default=str)[:12000])

