"""Ordinary SIFT + PnP, RGB-D reprojection, illumination correction and quality gate.

No simulator object IDs, world meshes, change labels or true poses are accepted here.
Oracle interventions are performed by the evaluation harness via explicit projected arrays.
"""
import time
import cv2
import numpy as np
from .renderer import unproject, project, transform


def estimate_pose(ref_rgb, ref_depth, query_rgb, K, roi, params):
    started = time.perf_counter()
    cv2.setRNGSeed(20261021)
    mask = np.full(ref_depth.shape, 255, np.uint8)
    x0, y0, x1, y1 = roi
    mask[max(0,y0-8):y1+9, max(0,x0-8):x1+9] = 0
    mask[ref_depth <= 0] = 0
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.025)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY), mask)
    kq, dq = sift.detectAndCompute(cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY), None)
    stats = {"reference_keypoints": len(kr), "query_keypoints": len(kq),
             "ratio_matches": 0, "inliers": 0, "elapsed_ms": 0.}
    if dr is None or dq is None:
        return None, stats, {}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, dq, k=2)
    good = [a for pair in pairs if len(pair)==2 for a,b in [pair]
            if a.distance < params["ratio_test"]*b.distance]
    uv_ref = np.array([kr[m.queryIdx].pt for m in good], np.float64).reshape(-1, 2)
    uv_q = np.array([kq[m.trainIdx].pt for m in good], np.float64).reshape(-1, 2)
    stats["ratio_matches"] = len(good)
    matches = {"ref": uv_ref, "query": uv_q, "inlier_mask": np.zeros(len(good), bool)}
    if len(good) < 6:
        return None, stats, matches
    xy = np.round(uv_ref).astype(int)
    z = ref_depth[xy[:, 1], xy[:, 0]]
    points = unproject(uv_ref, z, K)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, uv_q, K, None, iterationsCount=2000,
        reprojectionError=params["ransac_reprojection_px"], confidence=.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:
        return None, stats, matches
    inds = inliers.ravel()
    if len(inds) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(points[inds], uv_q[inds], K, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    repro = project(transform(points[inds], T), K)
    stats.update(inliers=len(inds),
                 median_reprojection_px=float(np.median(np.linalg.norm(repro-uv_q[inds], axis=1))),
                 elapsed_ms=1000*(time.perf_counter()-started))
    matches["inlier_mask"][inds] = True
    if len(inds) < params["min_inliers"]:
        return None, stats, matches
    return T, stats, matches


def projected_reference(ref_depth, K, T, roi=None, stride=1):
    h, w = ref_depth.shape
    x0,y0,x1,y1 = [0,0,w-1,h-1] if roi is None else roi
    yy,xx = np.mgrid[y0:y1+1:stride,x0:x1+1:stride]
    uv = np.column_stack([xx.ravel(),yy.ravel()])
    z = ref_depth[uv[:,1],uv[:,0]]
    keep = z > 0
    uv,z = uv[keep],z[keep]
    p = transform(unproject(uv,z,K),T)
    return uv, project(p,K), p[:,2]


def sample(query_rgb, query_depth, projected_uv, expected_depth):
    h,w = query_depth.shape
    finite = np.isfinite(projected_uv).all(axis=1)
    rounded = np.round(np.nan_to_num(projected_uv)).astype(int)
    inside = (finite & (expected_depth>0) & (rounded[:,0]>=0) & (rounded[:,0]<w)
              & (rounded[:,1]>=0) & (rounded[:,1]<h))
    xy = rounded.copy()
    xy[:,0] = np.clip(xy[:,0],0,w-1)
    xy[:,1] = np.clip(xy[:,1],0,h-1)
    colors = query_rgb[xy[:,1],xy[:,0]].astype(float)/255.
    depth = query_depth[xy[:,1],xy[:,0]]
    inside &= depth > 0
    return colors,depth,inside


def estimate_gain(ref_rgb, ref_depth, query_rgb, query_depth, K, T, roi, params):
    uv,pred,z = projected_reference(ref_depth,K,T,stride=3)
    colors,qz,inside = sample(query_rgb,query_depth,pred,z)
    x0,y0,x1,y1 = roi
    outside = ((uv[:,0]<x0-10)|(uv[:,0]>x1+10)|(uv[:,1]<y0-10)|(uv[:,1]>y1+10))
    rc = ref_rgb[uv[:,1],uv[:,0]].astype(float)/255.
    good = inside & outside & (np.abs(qz-z)<params["depth_registration_tolerance_m"])
    good &= (rc.min(axis=1)>.08)&(colors.min(axis=1)>.03)&(colors.max(axis=1)<.97)
    if good.sum()<100:
        return np.ones(3),int(good.sum())
    gain = np.clip(np.median(colors[good]/rc[good],axis=0),.25,2.)
    return gain,int(good.sum())


def assess(ref_rgb, ref_depth, query_rgb, query_depth, K, roi, T, params,
           robust=True, roi_projection=None):
    """Return measured evidence, no truth. roi_projection is reserved for labeled diagnostics."""
    if T is None:
        return {"decision":"insufficient","reason":"registration_failed","visible_fraction":0.,
                "changed_fraction":None,"gain":None}, {}
    uv,pred,z = projected_reference(ref_depth,K,T,roi)
    if roi_projection is not None:
        # True local correspondence diagnostic supplied explicitly by evaluator.
        uv,pred,z = roi_projection
    qc,qz,inside = sample(query_rgb,query_depth,pred,z)
    rc = ref_rgb[uv[:,1],uv[:,0]].astype(float)/255.
    gain, gain_n = estimate_gain(ref_rgb,ref_depth,query_rgb,query_depth,K,T,roi,params) if robust else (np.ones(3),0)
    color_residual = np.mean(np.abs(qc/gain-rc),axis=1)
    depth_residual = qz-z
    occluded = inside & (depth_residual < -params["occlusion_margin_m"])
    usable = inside & ~occluded if robust else inside
    changed = color_residual > params["color_residual_threshold"]
    if robust:
        changed |= depth_residual > params["missing_depth_margin_m"]
    count = int(usable.sum())
    fraction = count/max(1,len(uv))
    score = float(changed[usable].mean()) if count else None
    if count<params["min_roi_samples"] or fraction<params["min_visible_fraction"]:
        decision,reason="insufficient","visibility_or_sample_quality"
    else:
        decision="changed" if score>params["changed_fraction_threshold"] else "unchanged"
        reason="reprojected_rgbd_evidence"
    result={"decision":decision,"reason":reason,"roi_sample_count":len(uv),
            "usable_samples":count,"visible_fraction":float(fraction),
            "occluded_fraction":float(occluded.mean()),
            "changed_fraction":score,"gain":gain.tolist(),"gain_support_pixels":gain_n,
            "median_depth_residual_m":float(np.median(depth_residual[usable])) if count else None,
            "p95_depth_residual_m":float(np.percentile(depth_residual[usable],95)) if count else None}
    details={"reference_uv":uv,"projected_uv":pred,"expected_depth":z,
             "color_residual":color_residual,"depth_residual":depth_residual,
             "usable":usable,"occluded":occluded,"changed":changed,
             "sampled_query_rgb":np.clip(qc,0,1)}
    return result,details


def pose_errors(T_est,T_true):
    if T_est is None:
        return None,None
    relative=T_est[:3,:3]@T_true[:3,:3].T
    angle=np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
    return float(np.linalg.norm(T_est[:3,3]-T_true[:3,3])),float(angle)

