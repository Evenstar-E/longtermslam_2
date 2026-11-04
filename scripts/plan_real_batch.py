"""Freeze expanded pair list from metadata only, before running estimators."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
base=Path("data/external_feasibility_v1")
out=Path("data/real_batch_v1");out.mkdir(exist_ok=True)
if (out/"plan.json").exists():raise FileExistsError(out/"plan.json")
cfg=json.loads(Path("configs/real_batch_v1.json").read_text())
gt={};seq=None
for line in Path("third_party/openloris-scene-tools/benchmark/data/gt_office_1.txt").read_text().splitlines():
    if line.startswith("seq:"):seq=int(line.split(":")[1]);gt[seq]=[]
    elif line and line[0].isdigit():gt[seq].append([float(x) for x in line.split()])
gt={k:np.array(v) for k,v in gt.items()}
frames={}
for seq,folder in [(2,"index_v1"),(3,"index_v2"),(7,"index_v2")]:
    files=json.loads((base/folder/f"office1-{seq}.7z.json").read_text())["files"]
    prefix=np.cumsum([x["uncompressed"] for x in files])
    dep=[(i,f["name"],float(Path(f["name"]).stem)) for i,f in enumerate(files) if "/aligned_depth/" in f["name"]]
    rows=[]
    for i,f in enumerate(files):
        if "/color/" not in f["name"]:continue
        t=float(Path(f["name"]).stem);d=min(dep,key=lambda x:abs(t-x[2]))
        if abs(t-d[2])>cfg["selection"]["max_rgb_depth_gap_s"]:continue
        last=max(i,d[0])
        if prefix[last]>cfg["selection"]["max_uncompressed_prefix_mib"]*2**20:continue
        g=gt[seq][np.argmin(abs(gt[seq][:,0]-t))]
        if abs(g[0]-t)>cfg["selection"]["max_reference_gt_time_gap_s"]:continue
        rows.append({"sequence":seq,"rgb":f["name"],"depth":d[1],"rgb_timestamp":t,
            "depth_timestamp":d[2],"gt_timestamp":g[0],"base_pose":g[1:].tolist(),
            "prefix_bytes":int(prefix[last])})
    frames[seq]=rows
pairs=[];counts={}
for group in cfg["groups"]:
    candidates=[]
    for a in frames[group["reference"]]:
        p=np.array(a["base_pose"])
        for b in frames[group["current"]]:
            if b["rgb_timestamp"]<=a["rgb_timestamp"]:continue
            q=np.array(b["base_pose"])
            distance=float(np.linalg.norm(p[:3]-q[:3]))
            angle=float(2*np.degrees(np.arccos(np.clip(abs(np.dot(p[3:],q[3:])/(np.linalg.norm(p[3:])*np.linalg.norm(q[3:]))),0,1))))
            if distance>cfg["selection"]["max_translation_m"] or angle>cfg["selection"]["max_rotation_deg"]:continue
            stratum="near" if distance<=.5 and angle<=20 else "wider"
            candidates.append({"group":group["name"],"stratum":stratum,"reference":a,"current":b,
                "base_translation_gap_m":distance,"base_rotation_gap_deg":angle})
    counts[group["name"]]=dict(Counter(x["stratum"] for x in candidates))
    ref_used=Counter();cur_used=Counter();selected=[]
    for stratum in cfg["strata"]:
        pool=[x for x in candidates if x["stratum"]==stratum]
        # Deterministic temporal coverage, with low-I/O tie break and bounded frame reuse.
        for i in range(cfg["pairs_per_group_per_stratum"]):
            valid=[x for x in pool if ref_used[x["reference"]["rgb"]]<3 and cur_used[x["current"]["rgb"]]<3]
            if not valid:raise RuntimeError(f"Insufficient metadata candidates: {group['name']} {stratum}; {counts}")
            def score(x):
                v=np.array([x["reference"]["rgb_timestamp"],x["current"]["rgb_timestamp"]])
                if not selected:
                    return (-(x["reference"]["prefix_bytes"]+x["current"]["prefix_bytes"]),)
                gaps=[float(np.linalg.norm(v-np.array([s["reference"]["rgb_timestamp"],s["current"]["rgb_timestamp"]]))) for s in selected]
                return (min(gaps),-(x["reference"]["prefix_bytes"]+x["current"]["prefix_bytes"]))
            chosen=max(valid,key=score)
            chosen["pair_id"]=f"pair_{len(pairs):03d}"
            pairs.append(chosen);selected.append(chosen)
            ref_used[chosen["reference"]["rgb"]]+=1;cur_used[chosen["current"]["rgb"]]+=1
            pool.remove(chosen)
unique={f["rgb"]:f for x in pairs for f in (x["reference"],x["current"])}
result={"config_sha256":hashlib.sha256(Path("configs/real_batch_v1.json").read_bytes()).hexdigest(),
    "selection_before_inference":True,"metadata_candidates":counts,"pairs":pairs,
    "unique_frames":list(unique.values()),"independent_scenes":1,
    "scene":"OpenLORIS office","sequences":[2,3,7],"not_a_confirmation_set":True,
    "scope":"36 metadata-selected pairs; reused frames and correlated sequences;5 seeds are repeatability trials,not independent scenes."}
with (out/"plan.json").open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({"pairs":len(pairs),"unique_frames":len(unique),"candidates":counts,"planned_estimator_evaluations":len(pairs)*5*2}))

