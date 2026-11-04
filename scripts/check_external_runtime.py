"""Record read-only runtime and pinned-source readiness; does not install system software."""
import hashlib,json,platform,shutil,subprocess
from pathlib import Path
out=Path("results/external_feasibility_v1")
out.mkdir(parents=True,exist_ok=False)
def command(args,timeout=25):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,
                         creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        def decode(b):
            if b"\0" in b[:100]: return b.decode("utf-16-le",errors="replace").lstrip("\ufeff")
            return b.decode("utf-8",errors="replace")
        return {"argv":args,"returncode":p.returncode,"stdout":decode(p.stdout),"stderr":decode(p.stderr)}
    except Exception as e: return {"argv":args,"error":repr(e)}
repos={}
for name in ("Khronos","openloris-scene-tools","TorWICDataset"):
    p=Path("third_party")/name
    repos[name]={"head":command(["git","-C",str(p),"rev-parse","HEAD"]),
                 "latest_commit":command(["git","-C",str(p),"log","-1","--format=%H %cI %s"]),
                 "status":command(["git","-C",str(p),"status","--porcelain"])}
files=[
"third_party/Khronos/README.md","third_party/Khronos/LICENSE",
"third_party/Khronos/install/https.rosinstall",
"third_party/Khronos/khronos/CMakeLists.txt",
"third_party/Khronos/khronos_ros/launch/jackal_khronos.launch.yaml",
"third_party/Khronos/khronos_ros/launch/datasets/play_jackal.launch.yaml",
"third_party/Khronos/khronos_ros/config/datasets/jackal.yaml",
"third_party/openloris-scene-tools/benchmark/data/gt_office_1.txt",
"third_party/openloris-scene-tools/benchmark/openloris_tf_data.py",
"third_party/TorWICDataset/README.md","third_party/TorWICDataset/License/ReadMe.md"]
hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
check={"platform":platform.platform(),"disk_free_bytes":shutil.disk_usage(".").free,
       "host_commands":{name:shutil.which(name) for name in ("ros2","colcon","cmake","docker","wsl","git")},
       "docker":command(["docker","version","--format","{{json .}}"]),
       "wsl_list":command(["wsl","--list","--quiet"]),
       "wsl_ubuntu":command(["wsl","-d","Ubuntu-20.04","--","sh","-c",
            "cat /etc/os-release\ncommand -v ros2\ncommand -v colcon\ncommand -v cmake\nls -d /opt/ros/* 2>/dev/null"],timeout=30),
       "repos":repos,"selected_source_sha256":hashes,
       "torwic_cpp_files":[str(p) for pattern in ("*.cpp","*.h","CMakeLists.txt") for p in Path("third_party/TorWICDataset").rglob(pattern)],
       "torwic_python_files":[str(p) for p in Path("third_party/TorWICDataset").rglob("*.py")],
       "scope":"Read-only readiness; no full method build or SLAM result."}
with (out/"readiness.json").open("x",encoding="utf-8") as f: json.dump(check,f,ensure_ascii=False,indent=2)
print(json.dumps(check,ensure_ascii=False,indent=2))

