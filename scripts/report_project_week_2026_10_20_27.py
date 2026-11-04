"""Read-only retrospective of sealed experiments; does not train or run inference.

Run from repository root: .venv-real/Scripts/python.exe -m scripts.report_project_week_2026_10_20_27
Outputs are restricted to results/project_week_2026_10_20_27.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/project_week_2026_10_20_27"
VARIANTS = ["flat", "rich", "repeated", "changed"]
V16 = "results/material_partial_consensus_graph_v16"
SOURCES: dict[str, str] = {}
BLUE, ORANGE, GREEN, RED, GRAY = "#0072B2", "#E69F00", "#009E73", "#B44739", "#64748B"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(relative):
    path = ROOT / relative
    assert path.is_file(), relative
    SOURCES[str(Path(relative).as_posix())] = sha(path)
    return path


def read(relative):
    return json.loads(track(relative).read_text(encoding="utf-8-sig"))


def rows(relative):
    with track(relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(name, records):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def collect():
    data = read("data/textured_material_v1/audit.json")
    assert data["passed"] and data["scenes"] == 12 and data["sampled_correspondences"] == 135168
    v2 = read("data/textured_material_v2_train_probe/acquisition_budget.json")
    config = read("configs/textured_material_dataset_v2.json")
    assert config["splits"] == {"train": 1} and v2["full24"]["frames"] == 79
    current = read(f"{V16}/scores.json")
    tasks = rows(f"{V16}/task_metrics.csv")
    assert len(tasks) == current["primary_rows"] == 768
    for row in current["summary"]:
        subset = [r for r in tasks if (r["variant"], r["method"]) == (row["variant"], row["method"])]
        assert len(subset) == 24
        for name in ["qualified", "both_ids", "oracle_qualified"]:
            assert sum(int(r[name]) for r in subset) == row[name]
        assert sum(int(r["qualified"]) for r in subset if int(r["rear"])) == row["rear_qualified"]
    methods = {m: [next(r for r in current["summary"] if r["variant"] == v and r["method"] == m)
                   for v in VARIANTS] for m in dict.fromkeys(r["method"] for r in current["summary"])}
    chosen = methods["common_point_density"]
    qualified, identity = sum(r["qualified"] for r in chosen), sum(r["both_ids"] for r in chosen)
    assert (qualified, identity) == (60, 86)
    assert [r["qualified"] for r in methods["family_partial"]] == [14, 16, 14, 16]
    for comparison in current["comparisons"]:
        if comparison["baseline"] == "family_triplet" and comparison["candidate"] == "family_partial":
            assert not comparison["gained"] and not comparison["lost"]
    failures = Counter(r["category"] for r in rows(f"{V16}/failure_attribution.csv")
                       if r["method"] == "common_point_density")
    assert failures == {"success": 60, "no_qualified_pose_in_true_identity_pool": 30,
                        "true_identity_not_visited": 2, "objective_prefers_bad_pose_at_correct_identity": 2,
                        "identity_assignment_failed_despite_qualified_pose": 2}
    local = rows(f"{V16}/rich_local_seed_diagnosis.csv")
    support = Counter(int(r["true_pool_sources"]) for r in local)
    assert support == {0: 71, 1: 22, 2: 3} and len(local) == 96
    diagnosis = read(f"{V16}/diagnosis_summary.json")
    integrity = read(f"{V16}/completion_integrity.json")
    assert integrity["complete"] and integrity["old_evidence_unchanged"]
    old_path = "results/surface_measure_first_return_v1/scores.json"
    old = [r for r in read(old_path) if r["stage"] == "final"]
    assert SOURCES[old_path] == integrity["old_192_sha"]
    grouped = defaultdict(list)
    for r in old:
        grouped[(r["family"], r["method"])].append(r)
    old_summary = [{"family": f, "method": m, "tasks": len(rr), "qualified": sum(r["success"] for r in rr)}
                   for (f, m), rr in grouped.items()]
    selected = [("surface", "surface_independent"), ("surface", "surface_shared"),
                ("matched_voxel", "shared"), ("catalogue_plane", "typed_ray")]
    old_counts = [sum(r["success"] for r in grouped[key]) for key in selected]
    assert old_counts == [47, 79, 78, 120]
    assert all(len(grouped[key]) == 192 for key in selected)
    def key(row):
        return tuple(row[k] for k in ["scene", "geometry", "history_views", "asset"])
    def paired(a, b):
        aa, bb = [{key(r): bool(r["success"]) for r in grouped[k]} for k in [a, b]]
        assert aa.keys() == bb.keys() and len(aa) == 192
        return {"gained": sum(bb[i] and not aa[i] for i in aa), "lost": sum(aa[i] and not bb[i] for i in aa)}
    old_pairs = {"shared_vs_independent": paired(selected[0], selected[1]),
                 "shared_vs_matched_voxel": paired(selected[2], selected[1])}
    assert old_pairs == {"shared_vs_independent": {"gained": 36, "lost": 4},
                         "shared_vs_matched_voxel": {"gained": 11, "lost": 10}}
    training = []
    training_groups = ["material_joint_descriptor_training_v1", "material_joint_descriptor_training_v2",
                       "material_correspondence_training_v3", "material_appearance_training_v3",
                       "material_continuous_correspondence_training_v4"]
    excluded = []
    for group in training_groups:
        folder = f"results/{group}"
        seal = read(folder + "/training_seal.json")
        assert seal["complete"]
        for name, expected in seal["checkpoints"].items():
            path = track(folder + "/" + name)
            assert sha(path) == expected, path
            record = {"group": group, "checkpoint": name, "sha256": expected,
                      "source": folder + "/training_seal.json"}
            (excluded if name.startswith("untrained") else training).append(record)
    assert len(training) == 18 and len(excluded) == 2
    native = read("results/livingscenes_native_masks_v1/scores.json")
    real = read("results/livingscenes_real_v1/scores.json")
    native_summary = [r for r in native["summary"] if r["scene"] == "all"]
    real_summary = [r for r in real["summary"] if r["scene"] == "all"]
    pointdsc = read("results/pointdsc_material_reference_v15/scores.json")
    pointdsc_counts = [next(r["qualified"] for r in pointdsc["summary"]
                           if r["variant"] == v and r["method"] == "pointdsc_top1") for v in VARIANTS]
    assert pointdsc_counts == [13, 16, 14, 16]
    reports = [
        "direction_reassessment_2026-10-21.md", "expanded_real_experiment_2026-10-21.md",
        "khronos_reproduction_2026-10-21.md", "mainline_value_audit_2026-10-22.md",
        "projective_visibility_v1_2026-10-23.md", "se3_visibility_v1_2026-10-23.md",
        "real_cross_session_v1_2026-10-23.md", "rescan_author_execution_v1_2026-10-23.md",
        "rescan_third_period_results_v1_2026-10-24.md", "livingscenes_real_transfer_v1_2026-10-24.md",
        "novelty_core_decision_2026-10-24.md", "surface_measure_first_return_results_v1_2026-10-25.md",
        "textured_material_comparison_results_v1_2026-10-26.md", "material_joint_descriptor_results_v2_2026-10-26.md",
        "material_correspondence_repairs_v3_2026-10-26.md", "material_partial_consensus_results_v16_2026-10-27.md",
        "material_partial_consensus_model_v16_2026-10-27.md", "material_partial_consensus_decision_v16_2026-10-27.md"]
    for name in reports:
        track("research/" + name)
    for source in ["scripts/score_textured_material_comparison_v1.py", "scripts/score_material_native_modes_v6.py"]:
        track(source)
    result = {"period": ["2026-10-20", "2026-10-27"], "status": "ANALYZED: historical sealed outputs only",
              "date_boundary": "No Oct20 completion record found in local dated research/config/memory records; do not infer inactivity.",
              "dataset_v1": data, "dataset_v2_acquisition": v2, "trained_runs": len(training),
              "untrained_controls_excluded": excluded, "old192_summary": old_summary,
              "old192_paired": old_pairs, "current_summary": current["summary"],
              "current_comparisons": current["comparisons"], "current_selected_method": "common_point_density",
              "current_task_outcomes": {"qualified": qualified, "identity_correct_displacement_failed": identity-qualified,
                                        "identity_failed": 96-identity},
              "current_failure_attribution": dict(failures), "local_pool_true_sources": dict(support),
              "diagnosis": diagnosis, "living_native34": native_summary, "living_gt62": real_summary,
              "pointdsc_default": pointdsc_counts,
              "limitations": ["Old192 and current96 use different data/priors/contracts; no single performance trajectory.",
                              "Current96=24 assets x 4 correlated appearances in one previously seen TRAIN scene.",
                              "Local96=96 seed pools on GT-selected rear pairs, not 96 tasks.",
                              "No confidence intervals/p-values: this is descriptive and objects/appearances are correlated.",
                              "Published reference modules are not complete benchmark reproductions.",
                              "Check counts establish implementation consistency, not novelty or independent generalization."]}
    save_json("figure_source_data.json", result)
    save_csv("training_inventory.csv", training)
    save_csv("current_task_metrics.csv", tasks)
    save_csv("current_method_summary.csv", current["summary"])
    save_csv("old192_summary.csv", old_summary)
    ledger = []
    def metric(name, value, denominator, unit, source, scope):
        ledger.append(dict(metric=name, value=value, denominator=denominator, unit=unit, source=source,
                           source_sha256=SOURCES[source], interpretation=scope))
    for name in ["scenes", "asset_tasks", "frames", "pair_sets", "sampled_correspondences"]:
        metric("texture_v1_"+name, data[name], "", name, "data/textured_material_v1/audit.json", "Work volume; not independent sample count")
    for r in current["summary"]:
        metric(f"v16_{r['method']}_{r['variant']}_qualified", r["qualified"], 24, "asset-appearance tasks",
               f"{V16}/scores.json", "One seen TRAIN scene; four appearances correlated")
    for r in old_summary:
        metric(f"old192_{r['family']}_{r['method']}", r["qualified"], r["tasks"], "asset-condition tasks",
               old_path, "Familiar DEV regression; do not compare directly to current96")
    for r in training:
        metric("trained_checkpoint_"+r["group"]+"_"+r["checkpoint"], 1, "", "training artifact", r["source"],
               "SHA checked; excludes untrained controls; not independent training scenes")
    save_csv("metric_ledger.csv", ledger)
    return result


def style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    assert font.is_file()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})


def export(fig, name, pdf):
    fig.canvas.draw()
    # Do not crop inconsistently: all figures preserve their intended physical size.
    fig.savefig(OUT / (name + ".png"), dpi=190)
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".svg"))
    pdf.savefig(fig)
    plt.close(fig)


def bars(ax, labels, values, limit, colors, unit):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel(unit)
    ax.xaxis.grid(True, alpha=.16)
    ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(value + limit * .018, i, str(value), va="center", weight="bold", fontsize=12)


def figures(data):
    style()
    with PdfPages(OUT / "weekly_figures.pdf", metadata={"Title": "项目周报 2026-10-20—27", "Author": "Project retrospective"}) as pdf:
        fig = plt.figure(figsize=(16, 10))
        fig.text(.055, .95, "这一周：从巡检任务链路，推进到材料对应的核心瓶颈", fontsize=22, weight="bold", color="#102A43")
        fig.text(.055, .905, "2026.10.20—10.27  |  已完成的工作与证据边界  |  本次为汇总分析，未重新训练或跑实验", fontsize=12, color=GRAY)
        milestones = [
            ("10/21", "明确任务，打通原始数据", "OpenLORIS：36 对 / 57 帧 / 360 次方法评估；Khronos 官方示例运行", "重点转向：同一构件是否认对、变化是否量准"),
            ("10/22—23", "补齐视觉、几何和行动链路", "实际 RGB-D、完整 SE(3)、遮挡、观察/返航；接入 Rescan 与强配准方法", "遮挡修正使误报 12→0，但普通同条件方法也能做到"),
            ("10/24", "验证真实跨期与学习方法", "Rescan 三空间；两空间扩展第三期；运行 MORE² 作者模型及原生掩码对照", "新表面有用，错误身份会污染档案；学习特征并不自动更强"),
            ("10/25—26", "建立持续材料与首回波模型", "旧 192 项：连续表面、未见支持、射线曝光、连续位姿与共享/独立对照", "共享 79/192，独立 47/192；匹配体素 78/192，旧强方法 120/192"),
            ("10/26", "生成纹理数据并实际训练", "12 场景 / 135,168 对应；5 组训练共 18 个训练产物；全 24 资产近景探针", "能做几何/纹理/融合对比；局部误差改善尚未转成稳定任务增量"),
            ("10/27", "完整任务对照与失败定位", "全原生像素候选、PointDSC 参考、V16 四族生成；768 条完整方法记录", "身份正确 86/96，联合合格 60/96；V16 核心消融新增 0 项")]
        for i, (date, title, delivered, finding) in enumerate(milestones):
            y = .815 - i * .115
            fig.text(.055, y, date, color=BLUE, weight="bold", fontsize=15)
            fig.text(.18, y, title, weight="bold", fontsize=15)
            fig.text(.18, y-.032, delivered, fontsize=12)
            fig.text(.18, y-.061, finding, fontsize=11, color=GRAY)
        fig.text(.055, .085, "现在的位置：研究数据与后端比较链路已建立；新模型超过强对照及独立泛化，尚未证实。", fontsize=14, weight="bold", color=RED)
        fig.text(.055, .04, "10/20 未找到可核验的当日完成记录。各阶段分母不同，不累计为独立样本，也不连成一条性能提升曲线。", fontsize=11, color=GRAY)
        export(fig, "01_weekly_milestones", pdf)

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.subplots_adjust(left=.16, right=.97, top=.85, bottom=.19, wspace=.72, hspace=.66)
        fig.suptitle("量化结果：有局部收益，但核心任务仍有明显缺口", x=.06, ha="left", y=.97, fontsize=22, weight="bold")
        fig.text(.06, .925, "A 与 B–D 使用不同数据合同，不能直接排名；图中为描述性计数，无独立样本置信区间。", fontsize=12, color=GRAY)
        ax = axes[0, 0]
        bars(ax, ["连续表面·独立", "连续表面·共享", "匹配曝光体素", "旧强类型平面"], [47, 79, 78, 120], 192,
             [GRAY, BLUE, GREEN, ORANGE], "合格任务数 / 192")
        ax.set_title("A  旧 192 项：收益取决于对照", pad=16)
        ax.text(0, -.31, "共享比独立 +32；比匹配体素 +1。\n旧强方法先验不同，不能视为纯模块消融。", transform=ax.transAxes, fontsize=10, color=GRAY)
        ax = axes[0, 1]
        names = [("common_geometry", "普通几何", GRAY), ("common_descriptor_fusion", "普通几何+特征", BLUE),
                 ("family_partial", "V16 局部部分共识", ORANGE)]
        x = np.arange(4)
        for i, (method, name, color) in enumerate(names):
            counts = [next(r["qualified"] for r in data["current_summary"] if r["method"] == method and r["variant"] == v) for v in VARIANTS]
            positions = x + (i-1)*.24
            ax.bar(positions, counts, width=.22, color=color, label=name)
            for xx, count in zip(positions, counts): ax.text(xx, count+.35, str(count), ha="center", fontsize=9)
        ax.set_xticks(x, ["弱纹理", "丰富纹理", "重复纹理", "局部重涂"])
        ax.set_ylim(0, 24)
        ax.set_ylabel("合格任务数 / 每种外观 24")
        ax.set_title("B  当前全 24 资产：新机制未增加合格项", pad=16)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=1, frameon=False, fontsize=10)
        ax.yaxis.grid(True, alpha=.16); ax.set_axisbelow(True)
        ax = axes[1, 0]
        bars(ax, ["身份与位移均合格", "身份正确，位移失败", "身份失败"], [60, 26, 10], 96,
             [BLUE, ORANGE, RED], "资产×外观条件数 / 96")
        ax.set_title("C  认对身份，还不等于量准位移", pad=16)
        ax.text(0, -.30, "共同池普通点密度头：身份正确 86/96；\n联合合格 60/96（62.5%），背面 0/32。", transform=ax.transAxes, fontsize=10, color=GRAY)
        ax = axes[1, 1]
        bars(ax, ["正确身份池无合格位姿", "正确身份未进入搜索", "有好位姿但评分选错", "有好位姿但身份选错"], [30, 2, 2, 2], 36,
             [RED, ORANGE, GRAY, GRAY], "失败任务数 / 36")
        ax.set_title("D  32/36 失败在选择前已缺可用正确候选", pad=16)
        ax.text(0, -.30, "固定有限搜索池的事后诊断。\n不是原始图像没有信息的证明。", transform=ax.transAxes, fontsize=10, color=GRAY)
        fig.text(.06, .025, "B–D：同一个已见 TRAIN 场景，24 个资产 × 4 种相关外观；768 = 96 条件 × 8 方法，不能当作 768 个样本。", color=GRAY, fontsize=11)
        export(fig, "02_quantitative_results", pdf)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.subplots_adjust(left=.07, right=.96, top=.78, bottom=.20, wspace=.28)
        fig.suptitle("下一步该改哪里：先让真正的材料对应进入候选", x=.06, ha="left", y=.96, fontsize=22, weight="bold")
        fig.text(.06, .905, "独立诊断单位：rich 背面正确身份配对中的 96 个局部种子池；不同于上一图的 96 个任务条件。", fontsize=12, color=GRAY)
        ax = axes[0]
        counts = [71, 22, 3, 0]
        ax.bar(np.arange(4), counts, color=[RED, ORANGE, BLUE, GREEN], width=.58)
        for x, count in enumerate(counts): ax.text(x, count+1, str(count), ha="center", weight="bold", fontsize=14)
        ax.set_xticks(np.arange(4), ["0 个", "1 个", "2 个", "≥3 个"])
        ax.set_ylim(0, 96)
        ax.set_ylabel("局部候选池数 / 96")
        ax.set_xlabel("池中保留真实对应的不同源点数（1.5 cm 标准）")
        ax.set_title("A  71 个池没有真实材料对应", pad=18)
        ax.yaxis.grid(True, alpha=.16); ax.set_axisbelow(True)
        ax = axes[1]; ax.axis("off")
        blocks = [
            ("已证实", "当前池内：0/96 保留 ≥3 个真源点。\n新生成的背面身份配对仍 0/8 达标。", BLUE),
            ("机制并未转成收益", "容量约束在 91/96 个局部种子上起作用；\n但同起点 partial 对普通 TLS：0 新增、0 丢失。", ORANGE),
            ("下一步是信息形成，不是加迭代", "研究定向纹理/几何、位姿条件重检索、\n射线状态与未见支持；先对照普通定向匹配。", RED)]
        for i, (title, body, color) in enumerate(blocks):
            y = .88 - i*.32
            ax.text(0, y, title, fontsize=15, weight="bold", color=color, transform=ax.transAxes)
            ax.text(0, y-.16, body, fontsize=12, linespacing=1.7, transform=ax.transAxes)
        fig.text(.06, .09, "证据边界：这说明当前候选形成过程有问题；没有证明整个场景不可观测，也没有证明某一个新公式必然有效。", fontsize=11, color=GRAY)
        fig.text(.06, .045, "V17 尚未实现。应先取得同输入、同预算下的完整任务净增益，再冻结模型评估独立场景。", fontsize=12, weight="bold", color=BLUE)
        export(fig, "03_bottleneck_diagnosis", pdf)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = collect()
    figures(result)
    # Recheck every consumed historical input after reporting; no archived evidence was changed.
    assert all(sha(ROOT / path) == digest for path, digest in SOURCES.items())
    save_json("source_manifest.json", {"source_sha256": SOURCES, "script_sha256": sha(Path(__file__)),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "figures": {"dimensions_in": [[16, 10], [16, 11], [16, 8]], "png_dpi": 190,
                           "font": "Microsoft YaHei", "format": ["PNG", "PDF", "SVG"],
                           "purpose": "Internal weekly research briefing, not journal compliance certification"}})
    save_json("verification.json", {"complete": True, "historical_sources_rehashed": len(SOURCES),
               "trained_checkpoints_hash_verified": 18, "untrained_controls_hash_verified": 2,
               "old192_rows_compared_by_task_key": 192, "v16_primary_rows_reaggregated": 768,
               "v16_summary_groups_reconciled": 32, "local_pool_histogram_rows": 96,
               "historical_inputs_unchanged": True, "new_experiments": False,
               "figure_png_count": 3, "combined_pdf_pages": 3,
               "scope": "Evidence aggregation checks only; not new algorithm validation."})
    print(json.dumps({"output": str(OUT), "sources": len(SOURCES), "trained_runs": 18, "figures": 3,
                      "current_qualified": 60, "current_total": 96}, ensure_ascii=False))


if __name__ == "__main__":
    main()
