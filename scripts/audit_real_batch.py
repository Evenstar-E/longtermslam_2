"""Audit batch structure, source/raw integrity, pose evaluation and selected solver replays."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from PIL import Image
from scripts.real_batch_baselines import correspondence,depth_correspondences,pnp,rigid_ransac
from scripts.run_real_pair import calibration,extrinsic,pose
from visual_reinspection.baseline import pose_errors
from visual_reinspection.run import write_json
p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=a.run
manifest=json.loads((root/"manifest.json").read_text())
checks=[]
def add(name,passed):checks.append({"check":name,"passed":bool(passed)})
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for rel,value in manifest["source_sha256"].items():add("source:"+rel,sha(Path(rel))==value)
for rel,value in manifest["artifacts_sha256"].items():add("artifact:"+rel,sha(root/rel)==value)
cfg=json.loads((root/"config.json").read_text());plan=json.loads((root/"plan.json").read_text())
rows=[json.loads(x) for x in (root/"evaluations.jsonl").read_text().splitlines()]
add("360_complete_evaluations",len(rows)==360)
keys=[(x["pair_id"],x["method"],x["seed"]) for x in rows]
add("no_duplicate_trials",len(keys)==len(set(keys)))
add("36_unique_pairs",len({x["pair_id"] for x in rows})==36)
add("57_unique_frames",len(plan["unique_frames"])==57)
add("one_scene_explicit",plan["independent_scenes"]==1)
raw=Path(manifest["raw_input_dir"]);acq=json.loads((raw/"acquisition.json").read_text())
add("complete_120_raw_files",acq["complete"] and len(acq["records"])==120)
for f in acq["records"]:add("raw:"+f["path"],sha(raw/f["path"])==f["sha256"])
lookup={x["pair_id"]:x for x in plan["pairs"]}
cameras={s:extrinsic(raw/("office1-"+str(s))/"trans_matrix.yaml") for s in plan["sequences"]}
for row in rows:
    pair=lookup[row["pair_id"]]
    ref,cur=pair["reference"],pair["current"]
    true=np.linalg.inv(pose(cur["base_pose"])@cameras[cur["sequence"]])@(pose(ref["base_pose"])@cameras[ref["sequence"]])
    T=None if row.get("T_current_from_reference") is None else np.array(row["T_current_from_reference"])
    te,re=pose_errors(T,true)
    same=(te is None and row["translation_error_m"] is None) or (
        te is not None and np.isclose(te,row["translation_error_m"],rtol=1e-9,atol=1e-9)
        and np.isclose(re,row["rotation_error_deg"],rtol=1e-9,atol=1e-9))
    add("pose_evaluation:"+str((row["pair_id"],row["method"],row["seed"])),same)
    qualified=bool(te is not None and te<=.1 and re<=5)
    add("classification:"+str((row["pair_id"],row["method"],row["seed"])),
        row["qualified_pose"]==qualified and row["accepted_and_qualified"]==(row["accepted"] and qualified)
        and row["false_accept"]==(row["accepted"] and not qualified))
    if row["accepted"]:
        required=cfg["methods"][row["method"]]["min_final_inliers" if row["method"]=="pnp" else "min_inliers"]
        add("accepted_support:"+str((row["pair_id"],row["method"],row["seed"])),row["support"]>=required)
        add("proper_R:"+str((row["pair_id"],row["method"],row["seed"])),
            T is not None and np.isfinite(T).all() and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)
            and np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-8))
params=json.loads((root/"matching_parameters.json").read_text());cv2.setNumThreads(1)
for pair in (plan["pairs"][0],plan["pairs"][-1]):
    ref,cur=pair["reference"],pair["current"]
    r=np.array(Image.open(raw/ref["rgb"]).convert("RGB"));q=np.array(Image.open(raw/cur["rgb"]).convert("RGB"))
    rd=np.array(Image.open(raw/ref["depth"])).astype(np.float32)*.001
    qd=np.array(Image.open(raw/cur["depth"])).astype(np.float32)*.001
    K=calibration(raw/("office1-"+str(ref["sequence"]))/"sensors.yaml")[0]
    uv,quv=correspondence(r,rd,q,params)
    points,pa,pb,_=depth_correspondences(uv,quv,rd,qd,K)
    seed=cfg["seeds"][0]
    for method in ("pnp","rgbd"):
        T,stats=pnp(points,quv,K,cfg["methods"][method],seed) if method=="pnp" else rigid_ransac(pa,pb,cfg["methods"][method],seed)
        expected=next(x for x in rows if (x["pair_id"],x["method"],x["seed"])==(pair["pair_id"],method,seed))
        equal=(T is None and expected.get("T_current_from_reference") is None) or (
             T is not None and expected.get("T_current_from_reference") is not None and
             np.allclose(T,expected["T_current_from_reference"],rtol=1e-9,atol=1e-9))
        add("replay:"+pair["pair_id"]+":"+method,equal and stats["accepted"]==expected["accepted"] and stats["support"]==expected["support"])
summary=json.loads((root/"summary.json").read_text())
for method in ("pnp","rgbd"):
    rr=[x for x in rows if x["method"]==method]
    ss=summary["overall"][method]
    add("summary:"+method,ss["evaluations"]==len(rr) and ss["accepted"]==sum(x["accepted"] for x in rr)
        and ss["false_accept"]==sum(x["false_accept"] for x in rr)
        and ss["accepted_and_qualified"]==sum(x["accepted_and_qualified"] for x in rr))
result={"passed":all(x["passed"] for x in checks),"checks":checks,
        "scope":"Integrity,record/evaluator consistency and4 selected deterministic solver replays;not independent scene efficacy or novelty."}
write_json(a.output,result)
print(json.dumps({"passed":result["passed"],"checks":len(checks),"failed":[x for x in checks if not x["passed"]]}))
if not result["passed"]:raise RuntimeError("Batch audit failed")

