"""Scientific figures from stored observations and diagnostics; raw PNGs are preserved."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .renderer import scene


COLORS={"odometry_raw":"#D55E00","sift_rgbd_quality":"#0072B2","oracle_pose":"#009E73"}


def save(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=130)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def roi_patch(ref,roi):
    x0,y0,x1,y1=roi
    return ref[y0:y1+1,x0:x1+1]


def projected_box(ax,detail,color,label):
    if not detail:
        return
    uv=detail["projected_uv"]
    lo,hi=uv.min(axis=0),uv.max(axis=0)
    ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,edgecolor=color,linewidth=1.6,label=label))


def make_figures(cfg,ref,roi,T_ref,data,episode,episode_images,output):
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    out=output/"figures"
    out.mkdir()
    fig,axes=plt.subplots(3,3,figsize=(15,11.6),layout="constrained")
    axes=axes.ravel()
    axes[0].imshow(ref["rgb"])
    x0,y0,x1,y1=roi
    axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color="#0072B2",linewidth=2))
    axes[0].set_title("Reference session | explicit P3 fastener ROI")
    for ax,(case,frame,rec,details,_,_) in zip(axes[1:],data):
        ax.imshow(frame["rgb"])
        for method in ["odometry_raw","sift_rgbd_quality"]:
            projected_box(ax,details[method],COLORS[method],method)
        ax.set_title(case["id"]+"\nraw: "+rec["odometry_raw"]["decision"]+
                     " | SIFT+RGB-D: "+rec["sift_rgbd_quality"]["decision"],fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("One procedural 3D scene, two acquisition sessions and 8 development conditions\n"
                 "Orange: perturbed pose projection; blue: image-estimated projection. No independent-scene statistics.",fontsize=12)
    save(fig,out/"overview")
    for case,frame,rec,details,matches,_ in data:
        fig,axes=plt.subplots(2,4,figsize=(14,6.8),layout="constrained")
        axes[0,0].imshow(roi_patch(ref["rgb"],roi),interpolation="nearest")
        axes[0,0].set_title("Historical ROI (enlarged)")
        axes[1,0].imshow(frame["rgb"])
        projected_box(axes[1,0],details["sift_rgbd_quality"],"#0072B2","estimated")
        axes[1,0].set_title("Current full RGB")
        for col,name in enumerate(["odometry_raw","sift_rgbd_quality","oracle_pose"],1):
            d=details[name]
            if not d:
                axes[0,col].text(.1,.5,"Registration failed")
                axes[1,col].text(.1,.5,"No residual")
                continue
            h,w=y1-y0+1,x1-x0+1
            patch=np.full((h,w,3),.65)
            residual=np.full((h,w),np.nan)
            uv=d["reference_uv"]-np.array([x0,y0])
            patch[uv[:,1],uv[:,0]]=d["sampled_query_rgb"]
            residual[uv[:,1],uv[:,0]]=np.where(d["usable"],d["color_residual"],np.nan)
            axes[0,col].imshow(patch,interpolation="nearest")
            score=rec[name]["changed_fraction"]
            score_str="NA" if score is None else f"{score:.3f}"
            axes[0,col].set_title(f"{name}\n{rec[name]['decision']} | changed={score_str}")
            im=axes[1,col].imshow(residual,cmap=plt.get_cmap("magma").with_extremes(bad="#B0B0B0"),
                                 vmin=0,vmax=.35,interpolation="nearest")
            axes[1,col].set_title(f"RGB residual | usable {rec[name]['visible_fraction']:.2f}")
        for ax in axes.ravel():
            ax.set_axis_off()
        fig.colorbar(im,ax=axes[1,1:],shrink=.75,label="Mean absolute RGB residual [0,1]; gray=unusable")
        fig.suptitle(case["id"]+" | geometrically reprojected samples, no image enhancement\n"
                     "Robust decisions also use depth; color heatmaps alone do not define the decision.",fontsize=11)
        save(fig,out/(case["id"]+"_diagnostic"))
    # Correspondence lines from estimated SIFT inliers, not simulator correspondences.
    selected=next(row for row in data if row[0]["id"]=="wrong_bay_removed")
    _,frame,rec,details,matches,_=selected
    h,w=ref["rgb"].shape[:2]
    canvas=np.concatenate([ref["rgb"],frame["rgb"]],axis=1)
    fig,ax=plt.subplots(figsize=(13,5.4),layout="constrained")
    ax.imshow(canvas)
    indices=np.flatnonzero(matches["inlier_mask"])
    for idx in indices[::max(1,len(indices)//35)][:35]:
        p,q=matches["ref"][idx],matches["query"][idx]+[w,0]
        ax.plot([p[0],q[0]],[p[1],q[1]],color="#56B4E9",alpha=.7,lw=.7)
        ax.plot(p[0],p[1],"o",ms=2,color="#E69F00")
    ax.set_axis_off()
    ax.set_title(f"Ordinary SIFT + PnP | {len(indices)} inliers; up to 35 drawn\n"
                 "Reference task ROI excluded from feature registration; all raw matches saved.")
    save(fig,out/"feature_matches")
    # Top-down x-z view, metric geometry and executed camera path.
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for solid in scene(cfg,removed=True,occluder="narrow"):
        if solid.name=="floor":
            continue
        if solid.kind=="box":
            ax.add_patch(Rectangle((solid.center[0]-solid.half[0],solid.center[2]-solid.half[2]),
                                   2*solid.half[0],2*solid.half[2],color="#999999",alpha=.5))
    positions=np.array(episode["positions_world_m"])
    ax.plot(positions[:,0],positions[:,2],"o-",color="#0072B2",label="Executed camera path")
    for i,p in enumerate(positions):
        ax.annotate(str(i),(p[0]+.035,p[2]+.035))
    ax.scatter([.23],[.106],marker="*",s=90,color="#D55E00",label="P3 fastener location")
    ax.set(xlabel="World x [m]",ylabel="World z [m]",title="Top-down geometry and return",xlim=(-1.8,1.8),ylim=(-.5,3.1))
    ax.set_aspect("equal");ax.legend(fontsize=8)
    costs=[a["resource_spent_s"] for a in episode["trace"] if "resource_spent_s" in a]
    ax2.plot(range(1,len(costs)+1),costs,"o-",color="#0072B2",label="Elapsed modeled time")
    ax2.axhline(episode["budget_s"],color="#D55E00",ls="--",label="Budget")
    ax2.set(xlabel="Executed action index",ylabel="Modeled time [s]",ylim=(0,episode["budget_s"]*1.1),
            title="Kinematic time cost; no battery model")
    ax2.legend()
    save(fig,out/"episode_trajectory")
    fig,axes=plt.subplots(1,max(1,len(episode_images)),figsize=(12,4.7),layout="constrained",squeeze=False)
    for i,img in enumerate(episode_images):
        axes[0,i].imshow(img);axes[0,i].set_axis_off();axes[0,i].set_title(f"Executed capture {i}")
    fig.suptitle("Ordinary quality-triggered lateral re-observation; future images unavailable before capture")
    save(fig,out/"episode_captures")


def write_report(summary,checks,ep,output):
    rows=[]
    zh={"changed":"变化","unchanged":"未变化","insufficient":"证据不足"}
    for row in summary["cases"]:
        m=row["methods"]
        pose=m["sift_rgbd_quality"]["pose_translation_error_m"]
        pose_str="NA" if pose is None else f"{pose*1000:.2f}"
        rows.append("| "+row["case_id"]+" | "+" | ".join(zh[m[x]["decision"]] for x in
                    ["odometry_raw","sift_rgbd_quality","oracle_pose","oracle_roi_only"])+f" | {pose_str} |")
    source_note="程序化三维开发样例，1个场景、8个受控条件；不是8个独立样本，不计算统计显著性或工业成功率。"
    text=f"""# 两次采集的视觉复检样例结果

{source_note}

## 已执行内容

- 按冻结于首次运行前的场景配置生成透视RGB-D：历史构件、可读ID、紧固件几何和前景遮挡。
- 任务明确为P3指定紧固件；历史ROI由场景作者生成像素标注，基线不读取世界ROI。
- 普通SIFT匹配及PnP-RANSAC估计相对位姿；配准时排除任务ROI，避免依靠待检测变化完成配准。
- 原始先验投影与普通配准、全局光照校正、深度遮挡／缺失证据、质量门限比较。
- 真实相对位姿和仅ROI真值投影作为特权诊断；单刚体场景中二者相关，不能当作独立因果分解。
- 执行一次质量不足触发的预定义侧移、再拍摄和返航；这不是新规划方法。

## 结果逐项列出

| 条件 | 先验投影＋原始差分 | SIFT＋RGB-D＋质量门限 | 真值位姿诊断 | 仅ROI真值对应诊断 | SIFT相对平移误差mm |
|---|---|---|---|---|---|
{chr(10).join(rows)}

“变化／未变化”是算法输出；完整物理标签和是否正确见[summary.json](summary.json)。拒判不算任务完成。
原始先验是对真值变换施加公开规定的受控偏差，不是SLAM测量结果。当前普通前端是真正从图像估计，没有读取该真值变换。

## 闭环轨迹

- 最终判断：{zh[ep['final_decision']] if ep['final_decision'] else '未产生'}；物理状态：{zh[ep['physical_state_truth']]}。
- 采集次数：{sum(a['action']=='capture_and_estimate' for a in ep['trace'])}。
- 建模总耗时：{ep['total_time_s']:.3f}秒，预算{ep['budget_s']}秒；预算满足：{ep['within_budget']}。
- 返航路径碰撞检查：{ep['return_path_collision_free']}；成功返航：{ep['returned_home']}。
- 速度与采集时间是配置假设，未建模动力学、真实电池或人工响应时间。
- [完整动作记录](episode.json)；[轨迹图](figures/episode_trajectory.png)；[按执行顺序采集的图像](figures/episode_captures.png)。

## 机制验收

基础成像机制检查通过：{checks['passed']}。检查包含照明不改变几何／物理状态、物理移除改变深度和颜色、相机转动改变图像、机体／云台等量反向旋转保持相机图像及坐标系旋转合法性。
检查结果见[mechanism_checks.json](mechanism_checks.json)。这些是环境检查，不是新方法有效性证据。初版只实现开发样例，未声称完成16条件、多场景确认实验。

## 图像与可复核数据

- [全条件图像](figures/overview.png)；[图像估计的特征对应](figures/feature_matches.png)。
- 每个条件另有“历史ROI／重投影当前ROI／残差”诊断PNG和PDF。
- 原始图像和米制光轴深度在public；物体ID、真值深度和相机位姿在evaluator。
- diagnostics包含全部匹配、内点及每个ROI像素的颜色／深度残差、遮挡与有效性。
- 图像未做局部增强；ROI为便于查看使用最近邻显示放大，不增加细节。
- 渲染为程序化纹理、针孔相机、局部漫反射、球形紧固件代理；未模拟投射阴影、镜头模糊或真实桥梁材质。

## 如何解释

这是普通工程方法和环境机制的演示。不能用弱原始差分的失败证明需要新算法；应优先检查SIFT＋RGB-D基线和真值诊断之间是否仍有任务缺口。
不能将受控偏差、球形紧固件、清晰纹理和已知ROI上的结果外推为工业场景、细裂纹检测或长期SLAM表现。
下一步取决于逐图核查和强基线剩余误差，而不是增加合成难度直到得到优势。

## 复现

从项目目录执行：

    .\\.venv\\Scripts\\python.exe -m visual_reinspection.run --config configs/visual_sample_v1.json --output results/visual_sample_new_run

程序拒绝覆盖已有目录。依赖为requirements-visual-lock.txt。源码、配置和输出哈希见manifest.json。
算法依据：[OpenCV特征匹配](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html)、[PnP坐标与求解](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)。
"""
    (output/"REPORT.md").write_text(text,encoding="utf-8")

