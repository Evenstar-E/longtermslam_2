"""Established registration baselines for a frozen real-pair diagnostic batch."""
import time
import cv2
import numpy as np
from scripts.diagnose_real_pair_rgbd import fit,residual
from visual_reinspection.renderer import unproject,transform,project


def correspondence(ref,depth,query,params):
    mask=np.full(depth.shape,255,np.uint8);mask[:9,:9]=0;mask[depth<=0]=0
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
    kr,dr=sift.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),mask)
    kq,dq=sift.detectAndCompute(cv2.cvtColor(query,cv2.COLOR_RGB2GRAY),None)
    if dr is None or dq is None:return np.empty((0,2)),np.empty((0,2))
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr,dq,k=2)
    good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<params["ratio_test"]*b.distance]
    return (np.array([kr[x.queryIdx].pt for x in good],float).reshape(-1,2),
            np.array([kq[x.trainIdx].pt for x in good],float).reshape(-1,2))


def pnp(points,uv,K,cfg,seed):
    started=time.perf_counter()
    if len(points)<6:return None,{"accepted":False,"reason":"too_few_matches","support":0,"elapsed_ms":0.}
    cv2.setRNGSeed(seed)
    ok,r,t,ids=cv2.solvePnPRansac(points,uv,K,None,iterationsCount=cfg["iterations"],
          reprojectionError=cfg["reprojection_px"],confidence=cfg["confidence"],flags=cv2.SOLVEPNP_EPNP)
    if not ok or ids is None:return None,{"accepted":False,"reason":"ransac_failed","support":0,"elapsed_ms":1000*(time.perf_counter()-started)}
    initial=ids.ravel()
    if len(initial)>=6:r,t=cv2.solvePnPRefineLM(points[initial],uv[initial],K,None,r,t)
    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.ravel()
    if not np.isfinite(T).all():
        return None,{"accepted":False,"reason":"nonfinite_pose","support":0,"elapsed_ms":1000*(time.perf_counter()-started)}
    moved=transform(points,T)
    residual_px=np.linalg.norm(project(moved,K)-uv,axis=1)
    support=(moved[:,2]>0)&np.isfinite(residual_px)&(residual_px<cfg["reprojection_px"])
    accepted=bool(support.sum()>=cfg["min_final_inliers"])
    return T,{"accepted":accepted,"reason":"geometric_support" if accepted else "final_support_insufficient",
        "initial_support":len(initial),"support":int(support.sum()),
        "median_inlier_residual":float(np.median(residual_px[support])) if support.any() else None,
        "residual_units":"pixel","elapsed_ms":1000*(time.perf_counter()-started)}


def rigid_ransac(a,b,cfg,seed):
    started=time.perf_counter()
    if len(a)<3:return None,{"accepted":False,"reason":"too_few_depth_matches","support":0,"elapsed_ms":0.}
    rng=np.random.default_rng(seed);best=None
    for _ in range(cfg["iterations"]):
        ids=rng.choice(len(a),3,replace=False)
        area=min(np.linalg.norm(np.cross(a[ids[1]]-a[ids[0]],a[ids[2]]-a[ids[0]])),
                 np.linalg.norm(np.cross(b[ids[1]]-b[ids[0]],b[ids[2]]-b[ids[0]])))
        if area<cfg["triangle_double_area_min_m2"]:continue
        T=fit(a[ids],b[ids]);r=residual(a,b,T);inside=r<cfg["residual_m"]
        score=(int(inside.sum()),-float(np.median(r[inside])) if inside.any() else -float("inf"))
        if best is None or score>best[0]:best=(score,T,inside)
    if best is None:return None,{"accepted":False,"reason":"degenerate_matches","support":0,"elapsed_ms":1000*(time.perf_counter()-started)}
    if best[2].sum()<3:
        return None,{"accepted":False,"reason":"no_refittable_consensus","support":int(best[2].sum()),"elapsed_ms":1000*(time.perf_counter()-started)}
    T=fit(a[best[2]],b[best[2]])
    if not np.isfinite(T).all():
        return None,{"accepted":False,"reason":"nonfinite_pose","support":0,"elapsed_ms":1000*(time.perf_counter()-started)}
    r=residual(a,b,T);inside=r<cfg["residual_m"]
    accepted=bool(inside.sum()>=cfg["min_inliers"])
    return T,{"accepted":accepted,"reason":"geometric_support" if accepted else "final_support_insufficient",
        "support":int(inside.sum()),"median_inlier_residual":float(np.median(r[inside])) if inside.any() else None,
        "residual_units":"metre","elapsed_ms":1000*(time.perf_counter()-started)}


def depth_correspondences(uv,quv,rd,qd,K):
    xy=np.round(uv).astype(int);qxy=np.round(quv).astype(int)
    if not len(xy):return np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),np.zeros(0,bool)
    rz=rd[xy[:,1],xy[:,0]];qz=qd[qxy[:,1],qxy[:,0]]
    all_points=unproject(uv,rz,K)
    keep=(rz>0)&(qz>0)
    return all_points,all_points[keep],unproject(quv[keep],qz[keep],K),keep

