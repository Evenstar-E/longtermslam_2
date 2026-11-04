"""Audit raw integrity, calibration, saved inference and ordinary SE(3) fitting conventions."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scripts.run_real_pair import calibration,pose
from scripts.diagnose_real_pair_rgbd import fit,residual
from visual_reinspection.renderer import unproject,project,transform
from visual_reinspection.baseline import estimate_pose,assess
from visual_reinspection.run import write_json

root=Path.cwd();base=Path("results/real_pair_io_v1")
checks=[]
def add(name,ok):
    checks.append({"check":name,"passed":bool(ok)})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for run,key in [(base,"artifacts_sha256"),(Path("results/real_pair_rgbd_ransac_v1"),"artifact_sha256")]:
    manifest=json.loads((run/"manifest.json").read_text())
    for rel,h in manifest[key].items():add(str(run)+":"+rel,sha(run/rel)==h)
    if "source_sha256" in manifest:
        for rel,h in manifest["source_sha256"].items():add("source:"+rel,sha(Path(rel))==h)
raw=Path("data/external_feasibility_v1/raw_sample_v2")
acq=json.loads((raw/"acquisition.json").read_text())
add("acquisition_complete",acq["complete"])
for row in acq["files"]:add("raw:"+row["path"],sha(raw/row["path"])==row["sha256"])
plan=json.loads(Path("data/external_feasibility_v1/sample_plan.json").read_text())["selected"]
add("actual_multi_day_capture",plan["capture_time_gap_s"]>2*86400)
frames={}
for role in ("reference","current"):
    rgb=np.array(Image.open(base/"public"/role/"rgb.png"))
    dep=np.load(base/"public"/role/"depth_m.npy")
    meta=json.loads((base/"public"/role/"metadata.json").read_text())
    K=np.array(meta["K"])
    row=plan[role]
    original_depth=np.array(Image.open(raw/row["depth"]))
    add("rgb_original:"+role,sha(base/"public"/role/"rgb.png")==sha(raw/row["rgb"]))
    add("depth_conversion:"+role,np.array_equal(dep,original_depth.astype(np.float32)*.001))
    add("geometry_contract:"+role,rgb.shape==(480,848,3) and dep.shape==(480,848)
        and dep.dtype==np.float32 and np.isfinite(dep).all() and (dep>=0).all())
    add("sync:"+role,meta["sync_gap_s"]<.02)
    add("calibration_order:"+role,600<K[0,0]<620 and 600<K[1,1]<620 and 420<K[0,2]<440 and 240<K[1,2]<260)
    frames[role]=(rgb,dep,K)
ref,dep,K=frames["reference"];q,qd,Kq=frames["current"]
roi=json.loads((base/"public/task.json").read_text())["historical_roi_xyxy"]
params=json.loads((base/"baseline_parameters.json").read_text())
cv2.setNumThreads(1)
T,stats,matches=estimate_pose(ref,dep,q,K,roi,params)
expected=json.loads((base/"summary.json").read_text())
add("ordinary_pose_replay",T is None and expected["estimated_T_current_from_reference"] is None)
add("ordinary_matches_replay",stats["inliers"]==expected["registration"]["inliers"] and stats["ratio_matches"]==44)
result,_=assess(ref,dep,q,qd,K,roi,T,params)
add("ordinary_reject_replay",result==expected["ordinary_detector"])
# Independent synthetic known-transform check; no new scene-performance claim.
rng=np.random.default_rng(5)
a=rng.uniform([-1,-1,1],[1,1,3],size=(30,3))
angle=np.deg2rad(23);truth=np.eye(4)
truth[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
truth[:3,3]=[.2,-.1,.3]
b=transform(a,truth)
fitted=fit(a,b)
add("rigid_fit_known_transform",np.allclose(fitted,truth,atol=1e-10))
add("rigid_fit_proper_rotation",np.isclose(np.linalg.det(fitted[:3,:3]),1))
rvec,_=cv2.Rodrigues(fitted[:3,:3])
opencv_uv,_=cv2.projectPoints(a,rvec,fitted[:3,3],K,None)
add("project_agrees_with_opencv",np.allclose(project(b,K),opencv_uv.reshape(-1,2)))
rr=Path("results/real_pair_rgbd_ransac_v1")
data=np.load(rr/"matches_3d.npz")
summary=json.loads((rr/"summary.json").read_text())
estimated=np.array(summary["T_current_from_reference"])
r=residual(data["reference_points"],data["current_points"],estimated)
add("rgbd_residuals_readback",np.allclose(r,data["residual_m"]))
add("rgbd_geometric_inliers",np.array_equal(r<.05,data["inliers"]) and int((r<.05).sum())==13)
frozen=json.loads(Path("results/visual_sample_dev_v1/manifest.json").read_text())
add("prior_visual_sources_unchanged",all(sha(Path(p))==h for p,h in frozen["source_sha256"].items()))
out={"passed":all(c["passed"] for c in checks),"checks":checks,
     "scope":"Raw integrity, calibration, public replay and rigid-fit conventions. No new accuracy, novelty, full SLAM or generalized efficacy claim.",
     "raw_files":len(acq["files"]),"source_sha256":sha(Path(__file__))}
write_json(Path("results/external_feasibility_v1/validation.json"),out)
print(json.dumps({"passed":out["passed"],"checks":len(checks),"failed":[c for c in checks if not c["passed"]]},ensure_ascii=False))
if not out["passed"]:raise RuntimeError("Audit failed")

