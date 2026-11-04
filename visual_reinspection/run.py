"""Generate a small two-session visual development package. Never overwrites a run."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from .renderer import (intrinsics, look_at, render, historical_roi, path_clear,
                       project, transform)
from .baseline import estimate_pose, assess, projected_reference, pose_errors


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_observation(folder, frame, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame["rgb"]).save(folder/"rgb.png")
    np.save(folder/"depth_m.npy", frame["depth"])
    write_json(folder/"metadata.json", metadata)


def save_truth(folder, frame, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/"object_ids.npy", frame["object_ids"])
    np.save(folder/"depth_true_m.npy", frame["truth_depth"])
    write_json(folder/"truth.json", dict(metadata, object_names=frame["object_names"]))


def method_record(name, result, details, truth, case, T_used, T_true, stats=None):
    te,re=pose_errors(T_used,T_true)
    asset_fraction=None
    if details:
        p=np.round(details["projected_uv"]).astype(int)
        h,w=truth["object_ids"].shape
        valid=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)&details["usable"]
        if valid.any():
            ids=truth["object_ids"][p[valid,1],p[valid,0]]
            names=[truth["object_names"][int(i)] if i>=0 else "" for i in ids]
            assets=[x.split("_")[0] for x in names if x.startswith(("P2","P3","P4"))]
            if assets:
                asset_fraction=sum(x=="P3" for x in assets)/len(assets)
    physical="changed" if case["removed"] else "unchanged"
    emitted=result["decision"]!="insufficient"
    return dict(method=name,**result,pose_translation_error_m=te,pose_rotation_error_deg=re,
                target_asset_fraction=asset_fraction,physical_state_truth=physical,
                state_correct_if_reported=(result["decision"]==physical) if emitted else None,
                reported=emitted,registration=stats,
                information="privileged diagnostic" if name.startswith("oracle") else "public observations")


def evaluate_case(cfg, ref, query, roi, T_ref, T_query, case):
    K,params=intrinsics(cfg),cfg["baseline"]
    true=np.linalg.inv(T_query)@T_ref
    prior=true.copy()
    prior[0,3]+=case["pose_bias_x_m"]  # explicit controlled prior error, not measured SLAM
    estimated,stats,matches=estimate_pose(ref["rgb"],ref["depth"],query["rgb"],K,roi,params)
    local_oracle=projected_reference(ref["depth"],K,true,roi)
    methods=[
        ("odometry_raw",prior,False,None),
        ("sift_rgbd_quality",estimated,True,None),
        ("oracle_pose",true,True,None),
        ("oracle_roi_only",estimated,True,local_oracle)]
    records,details={},{}
    for name,T,robust,override in methods:
        result,d=assess(ref["rgb"],ref["depth"],query["rgb"],query["depth"],K,roi,T,params,robust,override)
        records[name]=method_record(name,result,d,query,case,T,true,stats if name=="sift_rgbd_quality" else None)
        details[name]=d
    return records,details,prior,estimated,stats,matches,true


def mechanism_checks(cfg,T_ref):
    clean=render(cfg,T_ref,add_sensor_noise=False)
    dim=render(cfg,T_ref,brightness=.6,add_sensor_noise=False)
    changed=render(cfg,T_ref,removed=True,add_sensor_noise=False)
    yaw=.37
    R=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    cancel=T_ref.copy()
    cancel[:3,:3]=T_ref[:3,:3]@R@R.T
    equivalent=render(cfg,cancel,add_sensor_noise=False)
    rotated=T_ref.copy()
    rotated[:3,:3]=T_ref[:3,:3]@R
    turned=render(cfg,rotated,add_sensor_noise=False)
    checks={
        "lighting_changes_rgb":bool(np.any(clean["rgb"]!=dim["rgb"])),
        "lighting_does_not_change_depth":bool(np.array_equal(clean["depth"],dim["depth"])),
        "lighting_does_not_change_instance_truth":bool(np.array_equal(clean["object_ids"],dim["object_ids"])),
        "physical_removal_changes_rgb_and_depth":bool(np.any(clean["rgb"]!=changed["rgb"]) and np.any(clean["depth"]!=changed["depth"])),
        "camera_rotation_changes_image":bool(np.mean(np.abs(clean["rgb"].astype(float)-turned["rgb"]))>1),
        "body_gimbal_cancel_preserves_image":bool(np.max(np.abs(clean["rgb"].astype(int)-equivalent["rgb"].astype(int)))<=1),
        "proper_camera_rotation":bool(np.isclose(np.linalg.det(T_ref[:3,:3]),1)),
        "changed_depth_pixel_count":int(np.sum(np.abs(clean["depth"]-changed["depth"])>.005))}
    required=[v for k,v in checks.items() if k!="changed_depth_pixel_count"]
    checks["passed"]=all(required)
    return checks


def episode(cfg,ref,roi,T_ref,output):
    """One ordinary quality-triggered retry, capturing only after executing each move."""
    ep,c=cfg["episode"],cfg["capture"]
    world=next(x for x in cfg["cases"] if x["id"]==ep["world_case"])
    home=np.array(ep["home_eye"],float)
    eye=home.copy()
    spent=0.
    trace=[]
    poses=[home.tolist()]
    observations=[]
    decision=None
    for index,destination in enumerate([home,np.array(c["side_eye"],float)]):
        if index:
            distance=float(np.linalg.norm(destination-eye))
            cost=distance/ep["speed_mps"]
            reserve=float(np.linalg.norm(destination-home))/ep["speed_mps"]
            if spent+cost+ep["capture_s"]+reserve>ep["budget_s"]:
                trace.append({"action":"stop","reason":"return_reserve"})
                break
            if not path_clear(cfg,[eye.tolist(),destination.tolist()],world["removed"],world["occluder"],ep["vehicle_radius_m"]):
                trace.append({"action":"stop","reason":"path_collision"})
                break
            spent+=cost
            trace.append({"action":"move_to_predeclared_side_view","distance_m":distance,"cost_s":cost,
                          "resource_spent_s":spent,"eye":destination.tolist()})
            eye=destination.copy()
            poses.append(eye.tolist())
        # No future image is produced or supplied before this capture action.
        T=look_at(eye,c["look_at"])
        frame=render(cfg,T,removed=world["removed"],brightness=world["brightness"],
                     occluder=world["occluder"],noise_seed=cfg["seed"]+100+index)
        spent+=ep["capture_s"]
        estimate,stats,_=estimate_pose(ref["rgb"],ref["depth"],frame["rgb"],intrinsics(cfg),roi,cfg["baseline"])
        result,_=assess(ref["rgb"],ref["depth"],frame["rgb"],frame["depth"],intrinsics(cfg),roi,estimate,cfg["baseline"])
        decision=result["decision"]
        save_observation(output/"public"/"episode"/f"frame_{index:03d}",frame,
                         {"timestamp_s":spent,"T_current_from_reference_estimated":estimate.tolist() if estimate is not None else None})
        save_truth(output/"evaluator"/"episode"/f"frame_{index:03d}",frame,{"T_world_camera":T.tolist()})
        trace.append({"action":"capture_and_estimate","frame_index":index,"cost_s":ep["capture_s"],
                      "resource_spent_s":spent,"result":result,"registration":stats})
        observations.append(frame["rgb"])
        if decision!="insufficient":
            break
    return_distance=float(np.linalg.norm(eye-home))
    return_cost=return_distance/ep["speed_mps"]
    clear=path_clear(cfg,[eye.tolist(),home.tolist()],world["removed"],world["occluder"],ep["vehicle_radius_m"])
    spent+=return_cost
    poses.append(home.tolist())
    trace.append({"action":"return_home","distance_m":return_distance,"cost_s":return_cost,"resource_spent_s":spent})
    result={"policy":ep["policy"],"trace":trace,"positions_world_m":poses,"total_time_s":spent,
            "budget_s":ep["budget_s"],"within_budget":spent<=ep["budget_s"],
            "return_path_collision_free":clear,"returned_home":clear and spent<=ep["budget_s"],
            "final_decision":decision,"physical_state_truth":"changed" if world["removed"] else "unchanged",
            "is_new_planner":False,"world_map_usage":"Predeclared camera path; full geometry used only for collision evaluator.",
            "evidence_scope":"Kinematic scripted retry, no autonomous exploration, SLAM, flight dynamics or calibrated power model."}
    write_json(output/"episode.json",result)
    return result,observations


def run(config_path,output):
    started=time.perf_counter()
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    cfg_bytes=config_path.read_bytes()
    (output/"config.json").write_bytes(cfg_bytes)
    T_ref=look_at(cfg["reference"]["eye"],cfg["reference"]["look_at"])
    K=intrinsics(cfg)
    roi=historical_roi(cfg,T_ref)
    ref=render(cfg,T_ref,noise_seed=cfg["seed"])
    save_observation(output/"public"/"reference",ref,{"K":K.tolist(),"roi_xyxy":roi,
                     "coordinate_frame":"reference camera; reference pose is identity by definition",
                     "depth_units":"metres along optical z","source":"procedural ray casting"})
    write_json(output/"public"/"task.json",{"asset_id":cfg["task"]["asset_id"],
               "surface":cfg["task"]["surface"],"question":cfg["task"]["question"],"historical_roi_xyxy":roi})
    save_truth(output/"evaluator"/"reference",ref,{"T_world_camera":T_ref.tolist()})
    summaries=[]
    visual_data=[]
    for i,case in enumerate(cfg["cases"]):
        c=cfg["capture"]
        eye=c["side_eye"] if case["view"]=="side" else c["eye"]
        T=look_at(eye,c["look_at"])
        frame=render(cfg,T,removed=case["removed"],brightness=case["brightness"],
                     occluder=case["occluder"],noise_seed=cfg["seed"]+i+1)
        records,details,prior,estimated,stats,matches,true=evaluate_case(cfg,ref,frame,roi,T_ref,T,case)
        folder=f"frame_{i:03d}"
        save_observation(output/"public"/"current"/folder,frame,
                         {"K":K.tolist(),"T_current_from_reference_odometry":prior.tolist(),
                          "odometry_source":"controlled perturbation; not output of a SLAM frontend"})
        save_truth(output/"evaluator"/"current"/folder,frame,
                   {"case":case,"T_world_camera":T.tolist(),"T_current_from_reference_true":true.tolist()})
        row={"case_id":case["id"],"frame":folder,"methods":records,
             "estimated_T_current_from_reference":estimated.tolist() if estimated is not None else None}
        write_json(output/"diagnostics"/(case["id"]+".json"),row)
        np.savez_compressed(output/"diagnostics"/(case["id"]+"_matches.npz"),**matches)
        for name,d in details.items():
            if d:
                np.savez_compressed(output/"diagnostics"/(case["id"]+"_"+name+".npz"),**d)
        summaries.append(row)
        visual_data.append((case,frame,records,details,matches,T))
        print(case["id"],{k:v["decision"] for k,v in records.items()},flush=True)
    checks=mechanism_checks(cfg,T_ref)
    ep,ep_images=episode(cfg,ref,roi,T_ref,output)
    write_json(output/"mechanism_checks.json",checks)
    summary={"schema_version":1,"source_type":"procedural_3d_development_sample","independent_scenes":1,
             "conditions":len(summaries),"not_a_statistical_benchmark":True,
             "config_sha256":hashlib.sha256(cfg_bytes).hexdigest(),"cases":summaries,
             "software":{"python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__},
             "elapsed_s":time.perf_counter()-started,
             "limitations":["No real images or photorealism validation","No trained or new method",
                            "No full 16-condition factorial study","Known historical image ROI; controlled pose prior",
                            "Texture-rich scene with legible IDs; image-derived single-rigid-scene pose estimation",
                            "Sphere-shaped fastener proxy; pinhole/Lambertian renderer lacks optics and cast shadows",
                            "Oracle local correspondences and pose are linked by one rigid scene model",
                            "No independent multiple-scene generalization evidence"]}
    write_json(output/"summary.json",summary)
    from .figures import make_figures,write_report
    make_figures(cfg,ref,roi,T_ref,visual_data,ep,ep_images,output)
    write_report(summary,checks,ep,output)
    source_paths=sorted(Path("visual_reinspection").glob("*.py"))+[config_path,Path("requirements-visual-lock.txt")]
    source_hashes={str(p).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    artifact_hashes={str(p.relative_to(output)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"manifest.json",{"source_sha256":source_hashes,"artifact_sha256":artifact_hashes})
    print("OUTPUT",output,"mechanism checks",checks["passed"],"episode",ep["final_decision"],flush=True)
    if not checks["passed"]:
        raise RuntimeError("Mechanism acceptance failed; outputs retained for diagnosis.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/visual_sample_v1.json"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.config,a.output)


if __name__=="__main__":
    main()

