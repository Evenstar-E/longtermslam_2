"""Post-hoc ordinary-solver diagnostic on identical saved correspondences; not a new method."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from visual_reinspection.renderer import unproject,transform,project
from visual_reinspection.baseline import pose_errors
from visual_reinspection.run import write_json
base=Path("results/real_pair_io_v1")
out=Path("results/real_pair_solver_diagnostic_v1")
out.mkdir(exist_ok=False)
ref=np.load(base/"public/reference/depth_m.npy")
query=np.load(base/"public/current/depth_m.npy")
K=np.array(json.loads((base/"public/reference/metadata.json").read_text())["K"])
true=np.array(json.loads((base/"evaluator/poses.json").read_text())["T_current_from_reference"])
m=np.load(base/"matches.npz")
uv=m["ref"];q=m["query"]
xy=np.round(uv).astype(int)
points=unproject(uv,ref[xy[:,1],xy[:,0]],K)
gt_res=np.linalg.norm(project(transform(points,true),K)-q,axis=1)
rows=[]
for name,flag in [("EPNP",cv2.SOLVEPNP_EPNP),("AP3P",cv2.SOLVEPNP_AP3P)]:
    cv2.setRNGSeed(20261021)
    ok,r,t,ids=cv2.solvePnPRansac(points,q,K,None,iterationsCount=2000,
                reprojectionError=2.5,confidence=.999,flags=flag)
    if not ok or ids is None:
        rows.append({"solver":name,"ok":False});continue
    inliers=ids.ravel()
    for refine in (False,True):
        rv,tv=(cv2.solvePnPRefineLM(points[inliers],q[inliers],K,None,r.copy(),t.copy())
               if refine else (r.copy(),t.copy()))
        T=np.eye(4);T[:3,:3]=cv2.Rodrigues(rv)[0];T[:3,3]=tv.ravel()
        residual=np.linalg.norm(project(transform(points,T),K)-q,axis=1)
        te,re=pose_errors(T,true)
        rows.append({"solver":name,"LM":refine,"ok":True,"initial_ransac_inliers":len(inliers),
          "recomputed_matches_under_2_5px":int((residual<2.5).sum()),
          "median_initial_inlier_residual_px":float(np.median(residual[inliers])),
          "translation_error_m":te,"rotation_error_deg":re,
          "T_current_from_reference":T.tolist()})
result={"kind":"posthoc_fixed_matches_standard_solver_diagnostic","matches":len(uv),
        "ordinary_solvers":rows,
        "gt_only_correspondence_check":{"under_2_5px":int((gt_res<2.5).sum()),
           "under_5px":int((gt_res<5).sum()),"under_10px":int((gt_res<10).sum()),
           "median_reprojection_px":float(np.median(gt_res))},
        "scope":"EPNP/AP3P and optional LM with same44 SIFT matches and fixed RANSAC settings. Ground truth only evaluates. No calibration, held-out confirmation or task accuracy.",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_json(out/"diagnostic.json",result)
print(json.dumps(result,indent=2))

