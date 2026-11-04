"""Select a small real cross-session pair using synchronization, pose overlap and I/O cost only."""
import json
from pathlib import Path
import numpy as np
base=Path("data/external_feasibility_v1")
gt={}
seq=None
for line in Path("third_party/openloris-scene-tools/benchmark/data/gt_office_1.txt").read_text().splitlines():
    if line.startswith("seq:"):
        seq=int(line.split(":")[1]);gt[seq]=[]
    elif line and line[0].isdigit():
        gt[seq].append([float(x) for x in line.split()])
gt={s:np.array(rows) for s,rows in gt.items()}
def pairs(seq,index):
    files=json.loads(index.read_text())["files"]
    rgb=[(i,x["name"],float(Path(x["name"]).stem)) for i,x in enumerate(files) if "/color/" in x["name"]]
    dep=[(i,x["name"],float(Path(x["name"]).stem)) for i,x in enumerate(files) if "/aligned_depth/" in x["name"]]
    prefix=np.cumsum([x["uncompressed"] for x in files])
    out=[]
    for i,name,t in rgb:
        d=min(dep,key=lambda x:abs(x[2]-t))
        if abs(d[2]-t)>.02: continue
        last=max(i,d[0])
        if prefix[last]>400*2**20: continue
        near=gt[seq][np.argmin(abs(gt[seq][:,0]-t))]
        if abs(near[0]-t)>.02: continue
        out.append({"sequence":seq,"rgb":name,"depth":d[1],"rgb_timestamp":t,
                    "depth_timestamp":d[2],"sync_gap_s":abs(d[2]-t),
                    "last_index":last,"uncompressed_prefix_bytes":int(prefix[last]),
                    "nearest_gt_timestamp":near[0],"base_pose":near[1:].tolist()})
    return out
a=pairs(2,base/"index_v1/office1-2.7z.json")
b=pairs(7,base/"index_v2/office1-7.7z.json")
candidates=[]
for x in a:
    p=np.array(x["base_pose"])
    for y in b:
        q=np.array(y["base_pose"])
        distance=float(np.linalg.norm(p[:3]-q[:3]))
        angle=float(2*np.degrees(np.arccos(np.clip(abs(np.dot(p[3:],q[3:])/(np.linalg.norm(p[3:])*np.linalg.norm(q[3:]))),0,1))))
        if distance>.5 or angle>20: continue
        cost=x["uncompressed_prefix_bytes"]+y["uncompressed_prefix_bytes"]
        candidates.append({"reference":x,"current":y,"base_translation_gap_m":distance,
            "base_rotation_gap_deg":angle,"summed_uncompressed_prefix_bytes":cost,
            "capture_time_gap_s":y["rgb_timestamp"]-x["rgb_timestamp"]})
if not candidates: raise RuntimeError("No bounded-I/O overlapping pair")
candidates.sort(key=lambda x:x["summed_uncompressed_prefix_bytes"])
result={"selection":"Minimize solid-block prefix I/O subject to RGB/depth gap<=20ms, GT support<=20ms, base translation<=0.5m and rotation<=20deg, per-sequence prefix<=400MiB.",
        "privileged_selection":"Official base_link ground truth used only for sample selection; not a navigation policy or detector input.",
        "no_performance_selection":True,"candidate_counts":[len(a),len(b)],"eligible_pairs":len(candidates),
        "selected":candidates[0],"next_best":candidates[1:5]}
with (base/"sample_plan.json").open("x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))

