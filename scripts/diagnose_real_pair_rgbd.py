"""Established 3D/3D rigid RANSAC on the same saved SIFT matches. Development diagnostic only."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from visual_reinspection.renderer import unproject
from visual_reinspection.baseline import assess,pose_errors
from visual_reinspection.run import write_json
from PIL import Image

SETTINGS={"seed":20261021,"iterations":2000,"residual_threshold_m":.05,
          "triangle_double_area_min_m2":1e-4,"min_inliers":6,
          "correspondences":"same44 SIFT ratio matches as first run; no learned matcher",
          "role":"posthoc ordinary 3D/3D diagnostic; query depth adds geometric evidence; not a new method"}

def fit(a,b):
    ac=a-a.mean(axis=0);bc=b-b.mean(axis=0)
    u,_,vt=np.linalg.svd(ac.T@bc)
    R=vt.T@u.T
    if np.linalg.det(R)<0:
        vt[-1]*=-1;R=vt.T@u.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=b.mean(axis=0)-R@a.mean(axis=0)
    return T
def residual(a,b,T):
    return np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)

def main():
    base=Path("results/real_pair_io_v1")
    out=Path("results/real_pair_rgbd_ransac_v1")
    out.mkdir(exist_ok=False)
    write_json(out/"settings.json",SETTINGS)
    rd=np.load(base/"public/reference/depth_m.npy")
    qd=np.load(base/"public/current/depth_m.npy")
    ref=np.array(Image.open(base/"public/reference/rgb.png"))
    query=np.array(Image.open(base/"public/current/rgb.png"))
    K=np.array(json.loads((base/"public/reference/metadata.json").read_text())["K"])
    true=np.array(json.loads((base/"evaluator/poses.json").read_text())["T_current_from_reference"])
    m=np.load(base/"matches.npz")
    uv,quv=m["ref"],m["query"]
    xy=np.round(uv).astype(int);qxy=np.round(quv).astype(int)
    rz=rd[xy[:,1],xy[:,0]];qz=qd[qxy[:,1],qxy[:,0]]
    keep=(rz>0)&(qz>0)
    a=unproject(uv[keep],rz[keep],K);b=unproject(quv[keep],qz[keep],K)
    rng=np.random.default_rng(SETTINGS["seed"])
    best=None
    for _ in range(SETTINGS["iterations"]):
        ids=rng.choice(len(a),3,replace=False)
        if min(np.linalg.norm(np.cross(a[ids[1]]-a[ids[0]],a[ids[2]]-a[ids[0]])),
               np.linalg.norm(np.cross(b[ids[1]]-b[ids[0]],b[ids[2]]-b[ids[0]])))<SETTINGS["triangle_double_area_min_m2"]:
            continue
        T=fit(a[ids],b[ids])
        r=residual(a,b,T);inside=r<SETTINGS["residual_threshold_m"]
        score=(int(inside.sum()),-float(np.median(r[inside])) if inside.any() else -float("inf"))
        if best is None or score>best[0]:best=(score,T,inside)
    if best is None:raise RuntimeError("No non-degenerate hypothesis")
    T=fit(a[best[2]],b[best[2]])
    r=residual(a,b,T);inside=r<SETTINGS["residual_threshold_m"]
    accepted=bool(inside.sum()>=SETTINGS["min_inliers"])
    roi=json.loads((base/"public/task.json").read_text())["historical_roi_xyxy"]
    params=json.loads((base/"baseline_parameters.json").read_text())
    result,details=assess(ref,rd,query,qd,K,roi,T if accepted else None,params)
    te,re=pose_errors(T,true)
    summary={"settings":SETTINGS,"usable_correspondences":len(a),
             "inliers":int(inside.sum()),"accepted":accepted,
             "median_3d_inlier_residual_m":float(np.median(r[inside])) if inside.any() else None,
             "T_current_from_reference":T.tolist(),
             "evaluation_translation_error_m":te,"evaluation_rotation_error_deg":re,
             "detector":result,"change_label":None,"accuracy":None,
             "scope":"One selected real pair, posthoc established solver diagnostic. No confirmation, defect labels or temporal SLAM.",
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out/"summary.json",summary)
    np.savez_compressed(out/"matches_3d.npz",reference_points=a,current_points=b,residual_m=r,inliers=inside,
                        original_match_keep=keep)
    if details:np.savez_compressed(out/"evidence.npz",**details)
    # Show the source correspondences actually supporting the 3D rigid estimate.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,5),layout="constrained")
    w=ref.shape[1];ax.imshow(np.concatenate([ref,query],axis=1))
    for p,q in zip(uv[keep][inside],quv[keep][inside]):
        ax.plot([p[0],q[0]+w],[p[1],q[1]],"-",lw=.8)
    ax.set_title("Ordinary 3D/3D rigid RANSAC: same SIFT matches; 5cm 3D residual gate; development only")
    ax.axis("off")
    fig.savefig(out/"matches.png",dpi=160);fig.savefig(out/"matches.pdf");plt.close(fig)
    artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out.rglob("*") if p.is_file()}
    write_json(out/"manifest.json",{"artifact_sha256":artifacts,"input_manifest_sha256":hashlib.sha256((base/"manifest.json").read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

