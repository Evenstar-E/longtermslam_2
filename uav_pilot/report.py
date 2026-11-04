"""Descriptive analysis and figures from unmodified, enumerated policy outcomes."""

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
from statistics import mean, median


AMBIGUOUS = {"fork", "observable", "pure_intent"}
LABELS = {"map": "MAP", "ask_first": "Ask first", "task_voi": "One-step task VOI",
          "lookahead_2": "Two-step lookahead", "lookahead_3": "Three-step lookahead", "exact": "Exact belief DP"}
COLORS = {"map": "#666666", "ask_first": "#0072B2", "task_voi": "#D55E00",
          "lookahead_2": "#CC79A7", "lookahead_3": "#E69F00", "exact": "#009E73"}
MARKERS = {"map": "s", "ask_first": "^", "task_voi": "D", "lookahead_2": "v", "lookahead_3": "P", "exact": "o"}


def interval(values, seed, repeats=2000):
    """Percentile bootstrap of independent layout means; descriptive, not population proof."""
    if not values:
        return {"n_layouts": 0, "mean": None, "low": None, "high": None}
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        return {"n_layouts": n, "mean": mean(values), "low": None, "high": None}
    samples = sorted(mean(rng.choices(values, k=n)) for _ in range(repeats))
    return {"n_layouts": n, "mean": mean(values),
            "low": samples[int(0.025 * (repeats-1))], "high": samples[int(0.975 * (repeats-1))]}


def paired_layout_difference(rows, baseline, metric, penalty, cfg):
    selected = [r for r in rows if r["status"] == "complete" and r["family"] in AMBIGUOUS
                and r["question_penalty"] == penalty]
    indexed = {(r["task_id"], r["method"]): r for r in selected}
    grouped = defaultdict(list)
    for row in selected:
        other = indexed.get((row["task_id"], baseline))
        if row["method"] == "exact" and other is not None:
            grouped[row["layout_id"]].append(row[metric] - other[metric])
    return interval([mean(v) for v in grouped.values()], cfg["seed"], cfg["bootstrap_repeats"])


def summarize(output: Path, cfg, manifest, rows):
    valid = [r for r in rows if r["status"] == "complete"]
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["family"], row["question_penalty"], row["method"])].append(row)
        if row["family"] in AMBIGUOUS:
            grouped[("ambiguous_only", row["question_penalty"], row["method"])].append(row)
    summary = []
    for (family, penalty, method), group in sorted(grouped.items()):
        summary.append({"family": family, "question_penalty": penalty, "method": method,
                        "n_tasks": len(group), "n_layouts": len({r["layout_id"] for r in group}),
                        **{key: mean(r[key] for r in group) for key in (
                            "success", "questions", "observations", "energy", "objective", "root_plan_ms")}})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    penalty = 20 if 20 in cfg["question_penalties"] else cfg["question_penalties"][0]
    differences = {baseline: {metric: paired_layout_difference(valid, baseline, metric, penalty, cfg)
                             for metric in ("objective", "success", "questions", "energy")}
                   for baseline in cfg["methods"] if baseline != "exact"}
    (output / "paired_differences.json").write_text(json.dumps({
        "direction": "exact minus baseline", "question_penalty": penalty,
        "population": "This synthetic layout generator; exploratory only",
        "replication_unit": "layout", "interval": "95% paired layout percentile bootstrap",
        "families": sorted(AMBIGUOUS), "results": differences,
    }, indent=2), encoding="utf-8")
    indexed = {(r["task_id"], r["question_penalty"], r["method"]): r for r in valid}
    worst = 0.0
    matched = 0
    for row in valid:
        exact = indexed.get((row["task_id"], row["question_penalty"], "exact"))
        if exact and row["method"] != "exact":
            worst = max(worst, exact["objective"] - row["objective"])
            matched += 1
    (output / "checks.json").write_text(json.dumps({
        "paired_objective_checks": matched, "max_exact_dominance_violation": worst,
        "tolerance": 1e-7, "all_passed": worst <= 1e-7,
        "note": "This consistency check does not prove continuous-flight optimality or algorithmic novelty.",
    }, indent=2), encoding="utf-8")
    headroom = []
    for penalty_value in cfg["question_penalties"]:
        for method in cfg["methods"]:
            cases = [r for r in valid if r["family"] in AMBIGUOUS and r["method"] == method
                     and r["question_penalty"] == penalty_value
                     and (r["task_id"], penalty_value, "exact") in indexed]
            if not cases:
                continue
            gaps = [r["objective"] - indexed[(r["task_id"], penalty_value, "exact")]["objective"] for r in cases]
            timings = sorted(r["root_plan_ms"] for r in cases)
            headroom.append({"method": method, "question_penalty": penalty_value,
                "n_tasks": len(cases), "positive_gap_tasks": sum(g > 1e-7 for g in gaps),
                "mean_objective_gap": mean(gaps), "max_objective_gap": max(gaps),
                "root_plan_median_ms": median(timings),
                "root_plan_p95_ms": timings[int(.95*(len(timings)-1))]})
    with (output / "headroom.csv").open("w", newline="", encoding="utf-8") as handle:
        if headroom:
            writer = csv.DictWriter(handle, fieldnames=list(headroom[0]))
            writer.writeheader()
            writer.writerows(headroom)
    main_rows = [r for r in summary if r["family"] == "ambiguous_only" and r["question_penalty"] == penalty]
    primary = cfg.get("primary_baseline", "task_voi")
    lines = [
        "# 离散规划预实验结果", "", "## Material Passport", "",
        "- Origin Mode: local implementation and simulation run",
        "- Verification Status: EXECUTED（离散模型结果；不代表实机验证）",
        f"- Version Label: {cfg.get('version_label', 'pilot_v1')}", "", "## 实际运行范围", "",
        f"- Split: {manifest['split']}；布局数 {manifest['independent_layouts']}；任务配置 {manifest['task_count']}。",
        f"- 成本权重 {cfg['question_penalties']}；策略数 {len(cfg['methods'])}；每条件重复 {cfg['rollout_repeats']}。",
        f"- 策略评估 {manifest['policy_evaluations']}；采样轨迹 {manifest['rollout_count']}；超时 {manifest['timeouts']}。",
        f"- 主计算运行时间 {manifest['elapsed_seconds']:.2f} 秒（不含绘图）。",
        "- 主表使用有限策略树全部观测分支的期望值；采样轨迹另外保存用于诊断。",
        "- 图路径预设安全，未模拟碰撞或动力学；返航约束成立不等于实机安全。",
        "", "## 主要工作点", "",
        f"目标 J = {cfg['failure_cost']} × 失败概率 + {cfg['energy_weight']} × 期望资源 + λq × 期望询问数。",
        f"下表固定 λq={penalty}，仅汇总 fork / observable / pure_intent，等权平均；不混入明确指令、等价目标或初始不可能任务。",
        "", "| 方法 | 期望成功率 | 询问/任务 | 观察/任务 | 资源/任务 | 目标 J（越低越好） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(f"| {LABELS[row['method']]} | {row['success']:.4f} | {row['questions']:.4f} | "
                     f"{row['observations']:.4f} | {row['energy']:.3f} | {row['objective']:.3f} |")
    lines += ["", "该表比较同一个预设目标函数。最小化该目标不保证单独成功率最高；应结合全部成本权重的工作点图阅读。", "",
              f"## 与 {LABELS[primary]} 的配对差异", "", "方向为精确策略减去指定基线。", ""]
    for metric, result in differences.get(primary, {}).items():
        if result["mean"] is None:
            continue
        ci = (f"[{result['low']:.4f}, {result['high']:.4f}]" if result["low"] is not None else "不估计（仅一个布局）")
        lines.append(f"- {metric}: {result['mean']:.4f}；95% 布局簇 bootstrap 区间 {ci}；布局数 {result['n_layouts']}。")
    lines += ["", "## 结论边界与后续", "",
        "本轮实现并比较的是已有基线与精确动态规划，没有提出或验证新的规划算法。精确策略的优势表示该离散设置中的改进空间，不能据此宣称新方法创新。",
        "主指标通过分支枚举消除了有限策略内的采样误差；跨布局区间只反映所用合成生成器的变化，不是工业场景泛化区间。",
        "headroom.csv 记录各方法相对精确策略的差距案例数、均值、最大值和根决策时延。通用多步方法若已消除差距，则当前设置缺乏专门算法的必要性证据。自然指令与廉价界面替代的需求审查仍未完成。",
        "", "## 实现边界", "",
        "- 两个符号解释、已知地图/先验/观测似然、可靠二元询问；未实现语言解析或 SE(3) 参考绑定。",
        "- 每观察点最多看一次；不同观察点的噪声在真值条件下独立。",
        "- finish 宏动作包含最短路到指定拍摄点、一次拍照、最短路返航；提交后不再获取信息。",
        "- 时域按宏决策次数计数；精确性仅针对该有限动作和时域模型，目标值比较容差 1e-7。",
        "- 两步/三步前瞻（如配置启用）在深度截断处使用 MAP 终端值，并在每次实际行动时重新前瞻；报告执行策略的完整期望，不报告截断预测值。",
        "- 搜索方法共享配置中的单策略超时上限；没有触及时不是等时截断试验。",
        "- 资源是合成单位；询问成本与额外人工打扰惩罚分别计入。",
        "- pre_capture_lost_recoverability 只检查拍摄前已执行路径上、初始真实任务可行的样本；不把任务提交错误自动等同于能源不可恢复。",
        "- root_plan_ms 是首次根决策的本机耗时；轨迹中的 policy_call_ms_cached 复用缓存，不是无人机在线端到端时延。",
        "- 非可行任务没有被删除，见 summary.csv 的 impossible 分组。",
        "", "## 文件与复现", "",
        "- config.json / tasks.jsonl：参数和全部公开任务定义。",
        "- expected.csv：每任务、成本权重、方法的精确期望指标；episodes.csv：采样结果。",
        "- traces.jsonl：所有轨迹，含仅用于评价的真值和资源诊断。",
        "- manifest.json：命令、版本、源文件与产物哈希；progress.json：进度。",
        "- paired_differences.json / checks.json：配对区间与最优值一致性检查。",
        "", "## 方法来源", "",
        "本项目是前期协议的最小实现。图表流程参考 scientific-visualization 技能，统计设计沿用预实验协议。",
        "方法知识库来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
        "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. "
        "[DOI](https://doi.org/10.48550/arXiv.2609.00065)。这不是 UAV 方法有效性的实验证据。", "",
    ]
    (output / "pilot_findings.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(output, cfg, manifest, rows, source_directory=None):
    source_directory = source_directory or output
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in rows if r["status"] == "complete"]
    descriptions = {}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
        for method in cfg["methods"]:
            operating_points = []
            for penalty in cfg["question_penalties"]:
                group = [r for r in valid if r["method"] == method and r["question_penalty"] == penalty
                         and r["family"] in AMBIGUOUS]
                if group:
                    operating_points.append((mean(r["questions"] for r in group),
                                             mean(r["energy"] for r in group),
                                             mean(r["success"] for r in group)))
            if not operating_points:
                continue
            for ax, column in zip(axes, (0, 1)):
                ax.plot([p[column] for p in operating_points], [p[2] for p in operating_points],
                        color=COLORS[method], marker=MARKERS[method], label=LABELS[method],
                        linewidth=1.6, markersize=10 if method == "map" else 6,
                        markerfacecolor="none" if method == "map" else COLORS[method],
                        zorder=5 if method == "map" else 3)
        axes[0].set(xlabel="Expected questions per task", ylabel="Expected joint task success",
                    xlim=(-0.03, 1.03), ylim=(-0.02, 1.03), title="Human interaction operating points")
        axes[1].set(xlabel="Expected resource cost (synthetic units)", ylabel="Expected joint task success",
                    ylim=(-0.02, 1.03), title="Resource operating points")
        for ax in axes:
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(cfg["methods"])), fontsize=9, frameon=False)
        fig.suptitle(f"Fixed synthetic suite: {manifest['independent_layouts']} layouts; ambiguous families only\n"
                     "Enumerated expectations; connected points vary question penalty (no interpolation claim)", fontsize=11)
        _save(fig, output, "operating_points")
        plt.close(fig)
        descriptions["operating_points"] = "Success versus questions and resource cost, averaged over the fixed ambiguous synthetic tasks. Points represent different question penalties. No industrial prevalence or generalization claim."

        penalty = 20 if 20 in cfg["question_penalties"] else cfg["question_penalties"][0]
        families = ["explicit", "equivalent", "fork", "observable", "pure_intent", "impossible"]
        families = [f for f in families if any(r["family"] == f for r in valid)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for offset, method in enumerate(cfg["methods"]):
            points = []
            for family in families:
                group = [r for r in valid if r["method"] == method and r["family"] == family
                         and r["question_penalty"] == penalty]
                points.append((mean(r["success"] for r in group), mean(r["questions"] for r in group))
                              if group else (float("nan"), float("nan")))
            width = 0.82 / len(cfg["methods"])
            positions = np.arange(len(families)) + (offset - (len(cfg["methods"])-1)/2) * width
            for ax, col in zip(axes, (0, 1)):
                ax.bar(positions, [p[col] for p in points], width=width * 0.94,
                       label=LABELS[method], color=COLORS[method], hatch=["", "//", "..", "xx", "--", "oo"][offset % 6])
        for ax, title in zip(axes, ("Expected joint success", "Expected questions")):
            ax.set(xticks=np.arange(len(families)), xticklabels=families, ylim=(0, 1.07), title=title)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(cfg["methods"])), fontsize=9, frameon=False)
        fig.suptitle(f"Diagnostic families at question penalty = {penalty}\n"
                     "Exact expectations on this finite suite; initially impossible tasks retained", fontsize=11)
        _save(fig, output, "family_comparison")
        plt.close(fig)
        descriptions["family_comparison"] = "Family-wise success and question count. Clear and equivalent tasks are negative controls. Initially impossible tasks remain visible; bars start at zero."

        if "lookahead_3" in cfg["methods"]:
            methods = ["task_voi", "lookahead_2", "lookahead_3", "exact"]
            cases = [r for r in valid if r["family"] in AMBIGUOUS and r["question_penalty"] == penalty]
            indexed = {(r["task_id"], r["method"]): r for r in cases}
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout="constrained")
            means, lows, highs, medians, p95s = [], [], [], [], []
            for method in methods:
                group = [r for r in cases if r["method"] == method and (r["task_id"], "exact") in indexed]
                layouts = defaultdict(list)
                for row in group:
                    layouts[row["layout_id"]].append(row["objective"] - indexed[(row["task_id"], "exact")]["objective"])
                estimate = interval([mean(v) for v in layouts.values()], cfg["seed"], cfg["bootstrap_repeats"])
                means.append(estimate["mean"])
                lows.append(estimate["low"] if estimate["low"] is not None else estimate["mean"])
                highs.append(estimate["high"] if estimate["high"] is not None else estimate["mean"])
                timing = sorted(r["root_plan_ms"] for r in group)
                medians.append(float(np.median(timing)))
                p95s.append(timing[int(.95*(len(timing)-1))])
            xs = np.arange(len(methods))
            axes[0].bar(xs, means, color=[COLORS[m] for m in methods], width=.55)
            axes[0].vlines(xs, lows, highs, color="black", linewidth=1.4)
            axes[0].scatter(xs, means, color="black", s=16, zorder=5)
            axes[0].set(ylabel="Expected objective gap to exact", ylim=(0, max(1, max(highs)*1.15)),
                        title="Remaining headroom\n95% paired-layout bootstrap intervals")
            axes[1].plot(xs, medians, "o-", color="#0072B2", label="Median")
            axes[1].plot(xs, p95s, "s--", color="#D55E00", label="95th percentile")
            axes[1].set(yscale="log", ylabel="First root decision (ms, log scale)",
                        title="Measured compute\nSame timeout cap; no equal-time truncation")
            axes[1].legend()
            for ax in axes:
                ax.set_xticks(xs, ["1-step", "2-step", "3-step", "Exact"])
                ax.grid(axis="y", alpha=.2)
            fig.suptitle(f"Generic lookahead comparison at question penalty {penalty}", fontsize=12)
            _save(fig, output, "lookahead_comparison")
            plt.close(fig)
            descriptions["lookahead_comparison"] = "Objective headroom to exact with paired layout bootstrap intervals, and uncached root-decision timing. Policies share a timeout cap; this is not equal-time truncation."

        if manifest["split"] == "sensitivity":
            tasks = [json.loads(line) for line in (source_directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            lookup = {t["task_id"]: t for t in tasks}
            budgets = sorted({t["budget"] for t in tasks})
            costs = sorted({t["ask_cost"] for t in tasks})
            indexed = {(r["task_id"], r["method"]): r for r in valid if r["question_penalty"] == penalty}
            data = np.full((len(costs), len(budgets)), np.nan)
            for task_id, task in lookup.items():
                if task["observation_rows"][0][1] != "19/20":
                    continue
                exact = indexed.get((task_id, "exact"))
                voi = indexed.get((task_id, "task_voi"))
                if exact and voi:
                    data[costs.index(task["ask_cost"]), budgets.index(task["budget"])] = voi["objective"] - exact["objective"]
            fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
            upper = max(1.0, float(np.nanmax(data))) if np.isfinite(data).any() else 1.0
            im = ax.imshow(data, cmap="cividis", vmin=0, vmax=upper,
                           aspect="auto", interpolation="nearest")
            ax.set(xticks=range(len(budgets)), xticklabels=budgets, yticks=range(len(costs)), yticklabels=costs,
                   xlabel="Initial resource budget at P", ylabel="Physical cost per question",
                   title=f"One-step VOI objective minus exact objective\nObservation reliability 0.95; question penalty {penalty}")
            for i in range(len(costs)):
                for j in range(len(budgets)):
                    if np.isfinite(data[i, j]):
                        color = "white" if data[i, j] < upper / 2 else "black"
                        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", color=color)
            fig.colorbar(im, ax=ax, label="Expected objective gap (synthetic units)")
            _save(fig, output, "sensitivity")
            plt.close(fig)
            descriptions["sensitivity"] = "One controlled layout: objective gap against one-step VOI across resource budgets and physical query costs, with fixed evidence quality. This is not an independent scene sample."
    (output / "figure_provenance.json").write_text(json.dumps({
        "source": str((source_directory / "expected.csv").resolve()),
        "transform": "Arithmetic means over named task families, no filtering of failed tasks",
        "uncertainty": "These plots show exact expectations of the fixed benchmark; layout-bootstrap differences are in paired_differences.json",
        "missing": "Timeout rows are absent, never converted to zero; manifest records count",
        "heatmap_scale": "Lower bound 0; upper bound max(1, observed maximum) to avoid an artificial signed scale for all-zero data",
        "formats": ["png at 180 dpi", "pdf vector"], "descriptions": descriptions,
    }, indent=2), encoding="utf-8")


def _save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180)
    fig.savefig(output / (name + ".pdf"))
