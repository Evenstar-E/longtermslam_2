"""Run an unchanged ordinary baseline on one actual OpenLORIS pair; no accuracy claim."""
import argparse,hashlib,json,platform,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from visual_reinspection.baseline import estimate_pose,assess,pose_errors
from visual_reinspection.run import write_json


def calibration(path):
    fs=cv2.FileStorage(path.read_text(encoding="utf-8"),cv2.FILE_STORAGE_READ|cv2.FILE_STORAGE_MEMORY)
    node=fs.getNode("d400_color_optical_frame")
    values=node.getNode("intrinsics").mat().ravel()
    # Official sensor file stores [fx,cx,fy,cy], NOT [fx,fy,cx,cy].
    fx,cx,fy,cy=values
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float)
    distortion=node.getNode("distortion_coefficients").mat().ravel()
    fs.release()
    if not np.allclose(distortion,0): raise ValueError("Requires explicit distortion handling")
    return K,values,distortion


def extrinsic(path):
    fs=cv2.FileStorage(path.read_text(encoding="utf-8"),cv2.FILE_STORAGE_READ|cv2.FILE_STORAGE_MEMORY)
    nodes=fs.getNode("trans_matrix")
    found=None
    for i in range(nodes.size()):
        node=nodes.at(i)
        if node.getNode("parent_frame").string()=="base_link" and node.getNode("child_frame").string()=="d400_color_optical_frame":
            found=node.getNode("matrix").mat()
    fs.release()
    if found is None: raise ValueError("No base/camera transform")
    return found


def pose(values):
    x,y,z,w=np.array(values[3:],float)/np.linalg.norm(values[3:])
    T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=values[:3]
    return T


def run(config,output):
    cfg=json.loads(config.read_text(encoding="utf-8"))
    plan=json.loads(Path(cfg["plan"]).read_text(encoding="utf-8"))["selected"]
    raw=Path(cfg["source_dir"])
    acquisition=json.loads((raw/"acquisition.json").read_text())
    if not acquisition["complete"]: raise RuntimeError("Incomplete data acquisition")
    for row in acquisition["files"]:
        if hashlib.sha256((raw/row["path"]).read_bytes()).hexdigest()!=row["sha256"]:
            raise RuntimeError("Raw file hash mismatch")
    params=json.loads(Path("configs/visual_sample_v1.json").read_text())["baseline"]
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config,output/"config.json")
    write_json(output/"baseline_parameters.json",params)
    frames={}
    public_meta={}
    true_poses={}
    for role in ("reference","current"):
        entry=plan[role]
        seq="office1-"+str(entry["sequence"])
        rgb=np.array(Image.open(raw/entry["rgb"]).convert("RGB"))
        raw_depth=np.array(Image.open(raw/entry["depth"]))
        if raw_depth.dtype!=np.uint16: raise ValueError("Expected original uint16 depth")
        depth=raw_depth.astype(np.float32)*cfg["depth_scale_m_per_unit"]
        K,original,distortion=calibration(raw/seq/"sensors.yaml")
        if rgb.shape!=(480,848,3) or depth.shape!=(480,848): raise ValueError("Shape mismatch")
        if not np.isfinite(depth).all() or (depth<0).any(): raise ValueError("Invalid depth")
        camera_to_base=extrinsic(raw/seq/"trans_matrix.yaml")
        true_poses[role]=pose(entry["base_pose"])@camera_to_base
        frames[role]=(rgb,depth,K)
        dest=output/"public"/role
        dest.mkdir(parents=True)
        shutil.copyfile(raw/entry["rgb"],dest/"rgb.png")
        np.save(dest/"depth_m.npy",depth)
        public_meta[role]={"K":K.tolist(),"original_intrinsics_order_fx_cx_fy_cy":original.tolist(),
            "distortion":distortion.tolist(),"original_shape":list(rgb.shape),
            "rgb_timestamp":entry["rgb_timestamp"],"depth_timestamp":entry["depth_timestamp"],
            "sync_gap_s":entry["sync_gap_s"],"valid_depth_fraction":float((depth>0).mean()),
            "depth_percentiles_m":np.percentile(depth[depth>0],[1,50,99]).tolist(),
            "depth_units":"metres along optical axis;0 invalid","rgb_unchanged_from_archive":True}
        write_json(dest/"metadata.json",public_meta[role])
    ref,dep,K=frames["reference"]
    query,qdep,Kq=frames["current"]
    if not np.allclose(K,Kq): raise ValueError("Current implementation requires identical K")
    roi=cfg["task"]["historical_roi_xyxy"]
    write_json(output/"public/task.json",cfg["task"])
    cv2.setNumThreads(1)
    # This call receives no pose, map, object labels or ground truth.
    T,stats,matches=estimate_pose(ref,dep,query,K,roi,params)
    result,details=assess(ref,dep,query,qdep,K,roi,T,params)
    true=np.linalg.inv(true_poses["current"])@true_poses["reference"]
    oracle,oracle_details=assess(ref,dep,query,qdep,K,roi,true,params)
    te,re=pose_errors(T,true)
    summary={"kind":cfg["kind"],"independent_pairs":1,"performance_benchmark":False,
       "capture_time_gap_s":plan["capture_time_gap_s"],"capture_time_gap_days":plan["capture_time_gap_s"]/86400,
       "public_metadata":public_meta,"registration":stats,
       "estimated_T_current_from_reference":None if T is None else T.tolist(),
       "evaluation_translation_error_m":te,"evaluation_rotation_error_deg":re,
       "ordinary_detector":result,"oracle_pose_detector":oracle,
       "change_label":None,"accuracy":None,
       "evidence_scope":"Input/calibration/ordinary-registration development smoke test, not defect detection or complete SLAM.",
       "adaptation":"Same estimator/detector code and parameters as procedural sample. Only actual calibration,depth conversion and analyst annotated ROI added.",
       "gt_scope":"GT selects overlap and evaluates pose; nearest measured base pose, camera extrinsic; not perfect truth; no GT supplied to ordinary estimator.",
       "software":{"python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__}}
    write_json(output/"summary.json",summary)
    write_json(output/"evaluator/poses.json",{"T_world_reference_camera":true_poses["reference"].tolist(),
        "T_world_current_camera":true_poses["current"].tolist(),"T_current_from_reference":true.tolist(),
        "nearest_gt_time_gaps_s":{k:abs(plan[k]["nearest_gt_timestamp"]-plan[k]["rgb_timestamp"]) for k in ("reference","current")},
        "selection_used_gt":True,"physical_change_label":None})
    np.savez_compressed(output/"matches.npz",**matches)
    if details: np.savez_compressed(output/"ordinary_evidence.npz",**details)
    if oracle_details: np.savez_compressed(output/"oracle_evidence.npz",**oracle_details)
    figures(frames,roi,matches,details,oracle_details,output)
    source_files=[config,Path("configs/visual_sample_v1.json"),Path(__file__),
                  Path("visual_reinspection/baseline.py"),Path("visual_reinspection/renderer.py")]
    manifest={"source_sha256":{str(p).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "artifacts_sha256":{str(p.relative_to(output)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in output.rglob("*") if p.is_file()},
        "raw_acquisition_sha256":hashlib.sha256((raw/"acquisition.json").read_bytes()).hexdigest()}
    write_json(output/"manifest.json",manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def figures(frames,roi,matches,details,oracle_details,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder=output/"figures"
    folder.mkdir()
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    cmap=plt.get_cmap("viridis").copy()
    cmap.set_bad("#c0c0c0")
    for col,role in enumerate(("reference","current")):
        rgb,depth,_=frames[role]
        axes[0,col].imshow(rgb)
        axes[0,col].set_title(role+" RGB (unmodified pixels)")
        image=axes[1,col].imshow(np.ma.masked_where(depth<=0,depth),cmap=cmap,vmin=.5,vmax=5)
        axes[1,col].set_title(role+" aligned depth (0 = invalid)")
        for row in (0,1): axes[row,col].axis("off")
    fig.colorbar(image,ax=list(axes[1,:]),label="Depth along optical axis [m]",shrink=.8)
    fig.suptitle("OpenLORIS-Scene: office1-2 / office1-7, 2.83 days apart; one selected pair")
    fig.savefig(folder/"real_pair.png",dpi=160)
    fig.savefig(folder/"real_pair.pdf")
    plt.close(fig)
    ref=frames["reference"][0];query=frames["current"][0]
    h,w=ref.shape[:2]
    fig,ax=plt.subplots(figsize=(14,5),layout="constrained")
    ax.imshow(np.concatenate([ref,query],axis=1))
    if matches:
        ids=np.flatnonzero(matches["inlier_mask"])
        for i in ids[np.linspace(0,len(ids)-1,min(35,len(ids)),dtype=int)] if len(ids) else []:
            a,b=matches["ref"][i],matches["query"][i]+[w,0]
            ax.plot([a[0],b[0]],[a[1],b[1]],lw=.65,alpha=.8)
    x0,y0,x1,y1=roi
    ax.add_patch(Rectangle((x0,y0),x1-x0+1,y1-y0+1,fill=False,edgecolor="orange",lw=2))
    ax.set_title("Ordinary SIFT/PnP inliers; orange historical task ROI excluded from registration")
    ax.axis("off")
    fig.savefig(folder/"inlier_matches.png",dpi=160)
    fig.savefig(folder/"inlier_matches.pdf")
    plt.close(fig)
    if not details:return
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout="constrained")
    axes[0,0].imshow(ref[y0:y1+1,x0:x1+1],interpolation="nearest")
    axes[0,0].set_title("Historical annotated ROI")
    shape=(y1-y0+1,x1-x0+1)
    # Valid historical depth can contain holes, so scatter values onto an ROI grid.
    for row,(label,data) in enumerate([("Estimated pose",details),("GT pose diagnostic",oracle_details)]):
        grid=np.full((*shape,3),.5)
        uv=data["reference_uv"]-[x0,y0]
        grid[uv[:,1],uv[:,0]]=data["sampled_query_rgb"]
        axes[row,1].imshow(grid,interpolation="nearest")
        axes[row,1].set_title(label+" warped RGB")
        residual=np.full(shape,np.nan)
        use=data["usable"]
        residual[uv[use,1],uv[use,0]]=data["color_residual"][use]
        cm=plt.get_cmap("magma").copy();cm.set_bad("#b0b0b0")
        im=axes[row,2].imshow(residual,vmin=0,vmax=.5,cmap=cm,interpolation="nearest")
        axes[row,2].set_title("RGB residual (gray = unusable)")
    axes[1,0].text(.05,.8,"One real pair; no change labels.\nNo detection accuracy measured.\nNearest-neighbor ROI display.\nDepth is also used by detector.",va="top",wrap=True)
    for ax in axes.flat:ax.axis("off")
    fig.colorbar(im,ax=list(axes[:,2]),label="Mean normalized RGB absolute residual",shrink=.7)
    fig.savefig(folder/"roi_diagnostic.png",dpi=160)
    fig.savefig(folder/"roi_diagnostic.pdf")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/real_pair_io_v1.json"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.config,a.output)

