"""Resolve official ROS1 dependency refs to immutable revisions; read-only remotes."""
import concurrent.futures,hashlib,json,subprocess
from pathlib import Path
import yaml
from visual_reinspection.run import write_json
root=Path("third_party/Khronos_ROS1")
source=root/"install/https.rosinstall"
entries=yaml.safe_load(source.read_text(encoding="utf-8"))
def resolve(entry):
    item=entry["git"].copy()
    version=item.get("version","HEAD")
    record={"name":item["local-name"],"uri":item["uri"],"declared_version":version}
    if len(version)==40 and all(c in "0123456789abcdef" for c in version):
        record.update(resolved_revision=version,status="official_declared_commit_not_fetched")
        item["version"]=version
        return item,record
    args=["git","-c","credential.helper=","ls-remote",item["uri"]]
    args+=["HEAD"] if version=="HEAD" else ["refs/heads/"+version,"refs/tags/"+version,"refs/tags/"+version+"^{}"]
    try:
        p=subprocess.run(args,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30,
                         creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        rows=[line.split() for line in p.stdout.splitlines() if line.strip()]
        if p.returncode or not rows:raise RuntimeError(p.stderr or "No matching public ref")
        # Peeled annotated tags preferred over tag object hashes.
        chosen=next((r for r in rows if r[1].endswith("^{}")),rows[0])
        item["version"]=chosen[0]
        record.update(status="resolved_remote_ref",resolved_revision=chosen[0],remote_ref=chosen[1])
    except Exception as e:
        record.update(status="unresolved",error=repr(e))
    return item,record
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    resolved=list(pool.map(resolve,entries))
out=Path("results/external_feasibility_v1")
records=[r for _,r in resolved]
locked=[{"git":item} for item,_ in resolved]
with (out/"khronos_ros1_locked.rosinstall").open("x",encoding="utf-8") as f:
    yaml.safe_dump(locked,f,sort_keys=False)
commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
write_json(out/"ros1_compatibility.json",{
"root_commit":commit,"official_manifest_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
"resolved_dependency_count":sum(r["status"]!="unresolved" for r in records),
"declared_commit_not_fetched_count":sum(r["status"]=="official_declared_commit_not_fetched" for r in records),
"dependencies":records,"host_linux":"WSL Ubuntu20.04.6, ROS Noetic",
"verified_linux_cmake":"3.16.3","verified_linux_eigen":"3.3.7-2","verified_linux_opencv":"4.2.0+dfsg-5",
"missing_or_not_found":["catkin command","ros-noetic-gtsam","libgtsam-dev","libgoogle-glog-dev","nlohmann-json3-dev"],
"build_attempted":False,
"scope":"Top-level dependency refs frozen; transitive manifests, apt packages, model weights and complete build not yet locked/verified."})
print(json.dumps({"root_commit":commit,"dependencies":len(records),
                 "unresolved":[r for r in records if r["status"]=="unresolved"]},ensure_ascii=False))

