"""Read-only run audit, public-input baseline replay and labeled post-hoc diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visual_reinspection.baseline import estimate_pose, assess


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected):
    if isinstance(actual, (str, bool)) or actual is None:
        return actual == expected
    return bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-10))


def audit(run, output):
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    manifest = read(run/"manifest.json")
    checks = []
    for kind, base in [("source_sha256", root), ("artifact_sha256", run)]:
        for rel, expected in manifest[kind].items():
            checks.append({"check": kind+":"+rel, "passed": sha(base/rel)==expected})
    # Only public observations and detector settings enter replay functions.
    public = run/"public"
    cfg = read(run/"config.json")
    params = cfg["baseline"]
    meta = read(public/"reference"/"metadata.json")
    K = np.array(meta["K"])
    roi = meta["roi_xyxy"]
    ref_rgb = np.array(Image.open(public/"reference"/"rgb.png"))
    ref_depth = np.load(public/"reference"/"depth_m.npy")
    public_files = list(public.rglob("*"))
    checks.append({"check":"public_file_contract", "passed": all(
        p.name in {"rgb.png","depth_m.npy","metadata.json","task.json"}
        for p in public_files if p.is_file())})
    summary = read(run/"summary.json")
    cv2.setNumThreads(1)
    replay = []
    sensitivity = []
    for row in summary["cases"]:
        folder = public/"current"/row["frame"]
        q_rgb = np.array(Image.open(folder/"rgb.png"))
        q_depth = np.load(folder/"depth_m.npy")
        prior = np.array(read(folder/"metadata.json")["T_current_from_reference_odometry"])
        checks.append({"check":"frame_shape:"+row["frame"],"passed": bool(
            q_rgb.shape==(480,640,3) and q_depth.shape==(480,640)
            and q_rgb.dtype==np.uint8 and q_depth.dtype==np.float32
            and np.isfinite(q_depth).all() and (q_depth>=0).all()
            and (q_depth>0).any())})  # zero denotes a ray with no surface hit
        T, stats, _ = estimate_pose(ref_rgb, ref_depth, q_rgb, K, roi, params)
        expected_T = row["estimated_T_current_from_reference"]
        pose_equal = (T is None and expected_T is None) or (
            T is not None and expected_T is not None and same(T,expected_T))
        checks.append({"check":"replayed_pose:"+row["case_id"],"passed":bool(pose_equal)})
        for method, transform, robust in [
            ("odometry_raw",prior,False),("sift_rgbd_quality",T,True)]:
            result, detail = assess(ref_rgb,ref_depth,q_rgb,q_depth,K,roi,transform,params,robust)
            expected = row["methods"][method]
            equal = all(same(value,expected[key]) for key,value in result.items())
            checks.append({"check":"replayed_evidence:"+row["case_id"]+":"+method,"passed":equal})
            replay.append({"case":row["case_id"],"method":method,"decision":result["decision"],
                           "all_detector_fields_match":equal})
            if method=="sift_rgbd_quality":
                total=len(detail["usable"])
                visible=int(detail["usable"].sum())
                changed=int((detail["changed"] & detail["usable"]).sum())
                gate = result["reason"]=="visibility_or_sample_quality"
                thresholds = {str(t):("insufficient" if gate else
                              "changed" if changed/visible>t else "unchanged")
                              for t in (.03,.05,.07,.10)}
                sensitivity.append({"case":row["case_id"],
                    "roi_samples":total,"usable_samples":visible,"changed_samples":changed,
                    "fraction_of_usable":changed/visible if visible else None,
                    "fraction_of_original_roi":changed/total,
                    "posthoc_threshold_decisions":thresholds})
    ep=read(run/"episode.json")
    checks.append({"check":"episode_cost_sum","passed":same(
        sum(t["cost_s"] for t in ep["trace"]),ep["total_time_s"])})
    checks.append({"check":"episode_return_and_budget","passed":bool(
        ep["returned_home"] and ep["within_budget"] and ep["return_path_collision_free"]
        and same(ep["positions_world_m"][0],ep["positions_world_m"][-1])
        and ep["total_time_s"]<=ep["budget_s"])})
    mechanisms=read(run/"mechanism_checks.json")
    checks.append({"check":"recorded_mechanism_checks","passed":mechanisms["passed"]})
    out={"passed":all(c["passed"] for c in checks),
         "source_files_verified":len(manifest["source_sha256"]),
         "artifacts_verified":len(manifest["artifact_sha256"]),
         "checks":checks,"public_input_replay":replay,
         "posthoc_sensitivity":sensitivity,
         "scope":"One procedural scene; post-hoc thresholds do not constitute calibration or confirmation.",
         "replay_scope":"Normal detectors replayed using saved public RGB-D, K, ROI, controlled odometry prior and baseline parameters. Oracle diagnostics not replayed.",
         "test_scope":"Functional and artifact consistency, not novelty, photorealism, industrial safety or generalization.",
         "auditor_sha256":sha(Path(__file__)),
         "run_manifest_sha256":sha(run/"manifest.json")}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:out[k] for k in ("passed","source_files_verified","artifacts_verified")}))
    for row in sensitivity:
        print(row)
    if not out["passed"]:
        raise RuntimeError("Audit failed; inspect output.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    audit(a.run,a.output)

