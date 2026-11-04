"""Execute frozen same-frame baseline comparisons; evaluation truth stays outside estimators."""
import argparse,csv,hashlib,json,platform,time
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np
from PIL import Image
from scripts.real_batch_baselines import correspondence,depth_correspondences,pnp,rigid_ransac
from scripts.run_real_pair import calibration,extrinsic,pose
from visual_reinspection.baseline import pose_errors
from visual_reinspection.run import write_json

def run(output,raw):
    cfg_path=Path("configs/real_batch_v1.json");cfg=json.loads(cfg_path.read_text())
    plan_path=Path("data/real_batch_v1/plan.json");plan=json.loads(plan_path.read_text())
    acq=json.loads((raw/"acquisition.json").read_text())
    if not acq["complete"]:raise RuntimeError("Incomplete input acquisition")
    if hashlib.sha256(cfg_path.read_bytes()).hexdigest()!=plan["config_sha256"]:raise RuntimeError("Frozen config changed")
    for row in acq["records"]:
        if hashlib.sha256((raw/row["path"]).read_bytes()).hexdigest()!=row["sha256"]:raise RuntimeError("Raw hash mismatch")
    output.mkdir(exist_ok=False,parents=True)
    (output/"config.json").write_bytes(cfg_path.read_bytes())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    params=json.loads(Path("configs/visual_sample_v1.json").read_text())["baseline"]
    write_json(output/"matching_parameters.json",params)
    cv2.setNumThreads(1)
    cameras={}
    for seq in plan["sequences"]:
        root=raw/("office1-"+str(seq))
        cameras[seq]=(calibration(root/"sensors.yaml")[0],extrinsic(root/"trans_matrix.yaml"))
    @lru_cache(maxsize=60)
    def image(rgb,depth):
        im=np.array(Image.open(raw/rgb).convert("RGB"))
        dep=np.array(Image.open(raw/depth))
        if im.shape!=(480,848,3) or dep.dtype!=np.uint16 or dep.shape!=(480,848):raise ValueError("Image contract")
        return im,dep.astype(np.float32)*.001
    rows=[];started=time.monotonic();match_stats=[]
    with (output/"evaluations.jsonl").open("x",encoding="utf-8") as stream:
        for pair in plan["pairs"]:
            a,b=pair["reference"],pair["current"]
            ref,rd=image(a["rgb"],a["depth"]);query,qd=image(b["rgb"],b["depth"])
            K,ext=cameras[a["sequence"]];Kq,extq=cameras[b["sequence"]]
            if not np.allclose(K,Kq):raise ValueError("Mismatched intrinsics")
            ts=time.perf_counter()
            uv,quv=correspondence(ref,rd,query,params)
            matching_ms=1000*(time.perf_counter()-ts)
            points,pa,pb,depth_keep=depth_correspondences(uv,quv,rd,qd,K)
            true=np.linalg.inv(pose(b["base_pose"])@extq)@(pose(a["base_pose"])@ext)
            match_stats.append({"pair_id":pair["pair_id"],"matches":len(uv),"depth_matches":len(pa),"matching_ms":matching_ms})
            for seed in cfg["seeds"]:
                for method in ("pnp","rgbd"):
                    try:
                        T,stats=(pnp(points,quv,K,cfg["methods"][method],seed) if method=="pnp" else
                                 rigid_ransac(pa,pb,cfg["methods"][method],seed))
                        te,re=pose_errors(T,true)
                        qualified=bool(T is not None and te<=cfg["evaluation"]["qualified_translation_m"]
                                              and re<=cfg["evaluation"]["qualified_rotation_deg"])
                        row={"pair_id":pair["pair_id"],"group":pair["group"],"stratum":pair["stratum"],
                             "method":method,"seed":seed,"matches":len(uv),"depth_matches":len(pa),
                             **stats,"translation_error_m":te,"rotation_error_deg":re,
                             "qualified_pose":qualified,"accepted_and_qualified":stats["accepted"] and qualified,
                             "false_accept":stats["accepted"] and not qualified,
                             "T_current_from_reference":None if T is None else T.tolist()}
                    except cv2.error as e:
                        row={"pair_id":pair["pair_id"],"group":pair["group"],"stratum":pair["stratum"],
                            "method":method,"seed":seed,"accepted":False,"qualified_pose":False,
                            "accepted_and_qualified":False,"false_accept":False,"error":str(e),
                            "translation_error_m":None,"rotation_error_deg":None}
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();rows.append(row)
            output.joinpath("monitor.json").write_text(json.dumps({"completed_pairs":len(match_stats),"total_pairs":len(plan["pairs"]),
                     "evaluations":len(rows),"elapsed_s":time.monotonic()-started}),encoding="utf-8")
            print(pair["pair_id"],pair["group"],pair["stratum"],"matches",len(uv),"done",len(rows),flush=True)
    summary={"kind":cfg["purpose"],"pairs":len(plan["pairs"]),"unique_frames":len(plan["unique_frames"]),
       "independent_scenes":1,"sequences":plan["sequences"],"random_seeds":cfg["seeds"],
       "evaluations":len(rows),"expected_evaluations":len(plan["pairs"])*len(cfg["seeds"])*2,
       "elapsed_s":time.monotonic()-started,"change_accuracy":None,
       "evaluation_contract":cfg["evaluation"],"groups":{},"overall":{},
       "limitations":["Single office scene; pairs and reused frames correlated","Seed repetitions measure solver variability,not new scenes",
          "No defect/change/task labels; coarse pose qualification only","Different solver support/residual gates",
          "Unchanged ordinary methods,no novel algorithm","GT used for metadata selection and offline evaluation only",
          "No Khronos results included in this ordinary-method table"]}
    def aggregate(selected):
        return {"evaluations":len(selected),"accepted":sum(x["accepted"] for x in selected),
          "accepted_and_qualified":sum(x["accepted_and_qualified"] for x in selected),
          "false_accept":sum(x["false_accept"] for x in selected),
          "accepted_translation_median_m":float(np.median([x["translation_error_m"] for x in selected if x["accepted"]])) if any(x["accepted"] for x in selected) else None,
          "accepted_rotation_median_deg":float(np.median([x["rotation_error_deg"] for x in selected if x["accepted"]])) if any(x["accepted"] for x in selected) else None}
    for method in ("pnp","rgbd"):
        summary["overall"][method]=aggregate([x for x in rows if x["method"]==method])
    for group in cfg["groups"]:
        name=group["name"]
        summary["groups"][name]={m:aggregate([x for x in rows if x["group"]==name and x["method"]==m]) for m in ("pnp","rgbd")}
    write_json(output/"summary.json",summary);write_json(output/"matching_stats.json",match_stats)
    from scripts.report_real_batch import figures
    figures(rows,plan,summary,output)
    sources=[cfg_path,Path(__file__),Path("scripts/real_batch_baselines.py"),Path("scripts/report_real_batch.py"),
             Path("scripts/run_real_pair.py"),Path("scripts/diagnose_real_pair_rgbd.py")]
    write_json(output/"manifest.json",{"raw_input_dir":str(raw),"source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "input_plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "acquisition_sha256":hashlib.sha256((raw/"acquisition.json").read_bytes()).hexdigest(),
        "artifacts_sha256":{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}})
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--data",type=Path,default=Path("data/real_batch_v1/raw"))
    a=p.parse_args();run(a.output,a.data)

