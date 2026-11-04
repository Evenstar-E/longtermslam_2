"""Validate and render a sourced DESK audit; never estimates interface efficacy."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


LABELS = {
    "interface_first": "先验证显式界面是否足够",
    "association_localization": "参考关联/重定位候选",
    "sensing_or_capability": "成像/物理检查能力",
    "resource_planning": "资源约束规划",
    "geometry_or_coverage": "几何/覆盖规划",
    "insufficient_context": "上下文不足，不能判断",
}
INTERFACES = {
    "pinned_frame": "固定截图/历史帧",
    "explicit_id": "显式编号",
    "point_selection": "点击目标/选面",
    "one_early_question": "起始一次询问",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/task_audit_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new output directory")
    sources = json.loads((args.input / "sources.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (args.input / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    indexed = {s["source_id"]: s for s in sources}
    require(len(indexed) == len(sources), "Duplicate source ID")
    require(len({c['case_id'] for c in cases}) == len(cases), "Duplicate case ID")
    for source in sources:
        require(urlparse(source["url"]).scheme == "https", "Source must have HTTPS URL")
        require(source["read_scope"] and source["does_not_support"], "Missing source limits")
    for c in cases:
        require(all(s in indexed for s in c["source_ids"]), "Unknown source")
        require(c["locator"] and c["context"]["missing"], "Missing locator/context gap")
        require(c["instruction"]["origin"] in {"researcher_reconstruction", "paper_example_paraphrase"}, "Unsupported natural-data claim")
        require(not c["instruction"]["verbatim"], "This audit stores paraphrases/reconstructions only")
        require(c["natural_operator_full_context_verified"] is False, "Unexpected full-context claim")
        require(c["primary_residual"] in LABELS, "Unknown classification")
        require(set(c["interfaces"]) == set(INTERFACES), "Missing interface counterfactual")
        require(all(v["status"] == "desk_inference_not_tested" and v["assessment"] for v in c["interfaces"].values()), "Unsupported empirical interface claim")
        require(all(v is None for v in c["empirical_interface_metrics"].values()), "Desk review cannot invent experimental metrics")
        require(len(c["candidate_interpretations_or_states"]) >= 2, "Missing contrasting states")
    summary = {
        "record_count": len(cases), "source_registry_count": len(sources),
        "sources_used_by_cases": len({s for c in cases for s in c["source_ids"]}),
        "origin_counts": dict(Counter(c["instruction"]["origin"] for c in cases)),
        "analyst_classification_counts": dict(Counter(c["primary_residual"] for c in cases)),
        "full_context_natural_operator_records": 0,
        "empirical_interface_trials": 0,
        "interpretation": "Counts describe analyst-created task units, not independent field deployments, prevalence, success rates or validated interface efficacy.",
    }
    lines = ["# 公开来源任务审查：16个任务单元", "",
        "这是可追溯的文档审查与反事实推断。指令均为重建/转述，未测界面效果。",
        "15条研究者重建 + 1条论文示例转述；0条完整原始工业操作者对话。不能把任务单元数当现场部署数。", "",
        "## 总览", "", "| ID | 任务单元 | 分析者初判（未实验） |", "|---|---|---|"]
    for c in cases:
        lines.append(f"| {c['case_id']} | {c['title']} | {LABELS[c['primary_residual']]} |")
    for c in cases:
        origin = "论文示例中文转述，非完整原始样本" if c["instruction"]["origin"] == "paper_example_paraphrase" else "研究者重建，非操作者原话"
        refs = ", ".join(f"[{s}: {indexed[s]['title']}]({indexed[s]['url']})" for s in c["source_ids"])
        lines += ["", f"## {c['case_id']} {c['title']}", "",
            f"来源事实：{c['evidence_summary']} {refs}", f"定位：{c['locator']}。", "",
            f"任务表达（{origin}）：{c['instruction']['text']}", "",
            f"已有上下文：{c['context']['provided']}", f"缺失材料：{c['context']['missing']}", "",
            "以下是本审查的反事实分析，不是来源报告的失败统计：", ""]
        lines += [f"- 候选解释/物理状态：{v}" for v in c["candidate_interpretations_or_states"]]
        lines += [f"- 行动后果：{c['action_consequence']}", "",
                  "| 接口 | 分析判断；效果未实测 |", "|---|---|"]
        lines += [f"| {label} | {c['interfaces'][key]['assessment']} |" for key, label in INTERFACES.items()]
        lines += ["", f"主瓶颈候选：{LABELS[c['primary_residual']]}；模块：{c['module_candidate']}。",
                  f"下一项测量：{c['next_measurement']}"]
        lines += [f"补充假设：{a}" for a in c["assumptions"]]
        lines += ["消歧率、交互耗时、任务成功率和能耗：均未测，原始记录为null。"]
    source_lines = ["# 来源与阅读范围", "", "所有访问日期均为2026-10-21。选读不等于全文系统审查；厂商文档不等于独立性能验证。", ""]
    for s in sources:
        source_lines += [f"## {s['source_id']} {s['title']}", "", f"[原始来源]({s['url']})", "",
            f"类型：{s['kind']}；来源标示日期：{s.get('date') or '未核实，勿以抓取日期代替'}。",
            f"阅读范围：{s['read_scope']}", f"可支持：{s['supports']}",
            f"不支持：{s['does_not_support']}",
            f"商业利益相关：{'是；功能与局限来自厂商自身文档' if s['commercial_interest'] else '未以厂商营销证据使用'}。", ""]
    args.output.mkdir(parents=True)
    outputs = {
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False),
        "casebook.md": "\n".join(lines), "sources.md": "\n".join(source_lines),
        "validation.json": json.dumps({"schema_and_evidence_boundary_checks": "passed", "case_count": len(cases),
            "note": "Checks validate record structure and explicit claim boundaries. They do not establish factual correctness, citation entailment or experimental efficacy."}, indent=2),
    }
    for name, body in outputs.items():
        (args.output / name).write_text(body + "\n", encoding="utf-8")
    inputs = [args.input / name for name in ("sources.json", "cases.jsonl", "search_log.json", "PROTOCOL.md")] + [Path(__file__)]
    manifest = {"input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "output_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in outputs},
        "scope": "offline record validation/rendering only; no new experiment or live source verification"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
