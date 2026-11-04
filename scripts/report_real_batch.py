"""Descriptive paired diagnostics; correlated pairs and seed repeats are labeled explicitly."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def figures(rows,plan,summary,output):
    folder=output/"figures";folder.mkdir()
    methods=["pnp","rgbd"];labels={"pnp":"SIFT + PnP (validated)","rgbd":"SIFT + 3D/3D RANSAC"}
    groups=list(summary["groups"])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    positions=np.arange(len(groups))
    for offset,method in [(-.18,"pnp"),(.18,"rgbd")]:
        data=[summary["groups"][g][method] for g in groups]
        good=np.array([d["accepted_and_qualified"]/d["evaluations"]*100 for d in data])
        wrong=np.array([d["false_accept"]/d["evaluations"]*100 for d in data])
        reject=100-good-wrong
        axes[0].bar(positions+offset,good,width=.34,color="#0072B2",label="Accepted + qualified" if offset<0 else None)
        axes[0].bar(positions+offset,wrong,bottom=good,width=.34,color="#D55E00",label="Accepted outside tolerance" if offset<0 else None)
        axes[0].bar(positions+offset,reject,bottom=good+wrong,width=.34,color="#CCCCCC",label="Rejected" if offset<0 else None)
    axes[0].set_xticks(positions,[g.replace("_"," ")+"\nPnP | 3D/3D" for g in groups],fontsize=8)
    axes[0].set_ylim(0,100);axes[0].set_ylabel("Fraction of solver evaluations [%]")
    axes[0].set_title("Coarse pose gate: <=10cm AND <=5deg");axes[0].legend(fontsize=8)
    for i,pair in enumerate(plan["pairs"]):
        ys=[]
        for method in methods:
            selected=[x for x in rows if x["pair_id"]==pair["pair_id"] and x["method"]==method]
            ys.append(np.mean([x["accepted_and_qualified"] for x in selected]))
        axes[1].plot([0,1],ys,lw=.7,alpha=.25,color="#555555")
        jitter=(i-(len(plan["pairs"])-1)/2)*.003
        axes[1].scatter(np.array([0,1])+jitter,ys,s=18,color=["#0072B2","#009E73"],alpha=.6)
    axes[1].set_xticks([0,1],[labels[m] for m in methods]);axes[1].set_ylim(-.04,1.04)
    axes[1].set_ylabel("Accepted + qualified fraction across five seeds")
    axes[1].set_title("Each line is one correlated image pair; no CI")
    fig.suptitle("OpenLORIS: 36 pairs / 57 frames / 3 sequences / 1 office scene; 360 evaluations")
    fig.savefig(folder/"comparison.png",dpi=180);fig.savefig(folder/"comparison.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,method in zip(axes,methods):
        means=[]
        for pair in plan["pairs"]:
            selected=[x for x in rows if x["pair_id"]==pair["pair_id"] and x["method"]==method]
            means.append(np.mean([x["accepted_and_qualified"] for x in selected]))
        image=ax.scatter([p["base_translation_gap_m"] for p in plan["pairs"]],
                    [p["base_rotation_gap_deg"] for p in plan["pairs"]],c=means,cmap="viridis",vmin=0,vmax=1,
                    s=55,edgecolors="black",linewidths=.4)
        ax.set_xlabel("Reference base translation gap [m]");ax.set_ylabel("Reference base rotation gap [deg]")
        ax.set_title(labels[method])
    fig.colorbar(image,ax=list(axes),label="Accepted + qualified fraction over five seeds")
    fig.suptitle("Descriptive geometry coverage; one scene, no independent-scene confidence interval")
    fig.savefig(folder/"geometry_coverage.png",dpi=180);fig.savefig(folder/"geometry_coverage.pdf");plt.close(fig)
    lines=["# 扩大后的真实图像配准诊断","","36对、57帧、3个采集序列、1个办公室场景；5个种子是求解重复，不是新场景。","",
      "配置与配对在运行前固定；近时间采集不自动标为物理未变化。没有缺陷／任务真值，没有巡检成功率。","",
      "粗位姿合格仅指相对官方参考误差≤0.10米且≤5度，不代表细缺陷对齐精度。","",
      "| 序列组 | 方法 | 评估数 | 接受且合格 | 接受但不合格 | 拒绝 |","|---|---|---:|---:|---:|---:|"]
    for g in groups:
        for m in methods:
            d=summary["groups"][g][m]
            lines.append(f"| {g} | {m} | {d['evaluations']} | {d['accepted_and_qualified']} | {d['false_accept']} | {d['evaluations']-d['accepted']} |")
    lines+=["","[结果总表](summary.json) · [逐次记录](evaluations.jsonl) · [固定配对](plan.json)",
      "","![描述性比较](figures/comparison.png)","","两种方法的支持量和残差门限不同，结果是开发诊断，不是通用排名或新方法证明。",
      "同一帧可能用于多对图像，多个序列仍来自同一个办公室；不做独立场景显著性检验。",
      "原单对实验的ROI被屏蔽，本轮是配准诊断，仅屏蔽图像左上角9×9像素；不能直接把两轮差异归因于样本数。",
      "PnP新增最终支持重算与正深度检查，这是普通输出验收；历史原始结果保持不变。",
      "Khronos完整时序系统未混入此表。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

