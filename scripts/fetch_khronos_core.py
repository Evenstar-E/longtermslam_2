"""Fetch only the pinned core dependencies needed by the ground-truth-input Khronos example."""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
import yaml
base=Path("third_party/khronos_ros1_deps")
base.mkdir(exist_ok=False)
out=Path("results/khronos_reproduction_v1")
out.mkdir(parents=True,exist_ok=False)
entries=yaml.safe_load(Path("results/external_feasibility_v1/khronos_ros1_locked.rosinstall").read_text())
needed={"catkin_simple","kimera_pgmo","kimera_rpgo","spark_dsg","hydra","hydra_ros",
        "pose_graph_tools","teaser_plusplus","spatial_hash","config_utilities"}
def fetch(e):
    item=e["git"];name=item["local-name"]
    if name not in needed:return {"name":name,"status":"optional_not_fetched"}
    dest=base/name
    record=dict(item);record["started"]=time.time()
    logs=[]
    commands=[["git","init",str(dest)],["git","-C",str(dest),"remote","add","origin",item["uri"]],
        ["git","-C",str(dest),"-c","credential.helper=","fetch","--depth=1","origin",item["version"]],
        ["git","-C",str(dest),"checkout","--detach","FETCH_HEAD"]]
    try:
        for cmd in commands:
            p=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=180,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            logs.append({"argv":cmd,"rc":p.returncode,"output":p.stdout+p.stderr})
            if p.returncode:raise RuntimeError("Git command failed")
        head=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
        if head!=item["version"]:raise RuntimeError("Wrong revision")
        record.update(status="fetched",head=head)
    except Exception as e:record.update(status="failed",error=repr(e))
    record["elapsed_s"]=time.time()-record["started"]
    (out/(name+"_fetch.json")).write_text(json.dumps(dict(record,commands=logs),indent=2),encoding="utf-8")
    print(name,record["status"],flush=True)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,entries))
(out/"source_fetch.json").write_text(json.dumps({"records":results,"all_core_fetched":all(
r["status"] in ("fetched","optional_not_fetched") for r in results)},indent=2),encoding="utf-8")

