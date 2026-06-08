"""评估报告生成 — Markdown 表格 + 分析结论。

面试时直接截图放进 PPT。
"""

from __future__ import annotations
from typing import Any


def format_eval_report(results: dict[str, Any]) -> str:
    """生成 Markdown 格式的评估报告。"""
    if "error" in results:
        return f"❌ 评估失败: {results['error']}"

    limit = results["limit"]
    lines = []

    lines.append(f"# 检索质量评估报告")
    lines.append(f"")
    lines.append(f"**知识库**: {results['paper_count']} 篇论文 | "
                 f"**查询数**: {results['query_count']} 条 | "
                 f"**截断**: k={limit}")
    lines.append(f"")

    # ── 汇总对比表 ──
    lines.append(f"## 汇总对比")
    lines.append(f"")
    headers = ["检索变体", f"Recall@{limit}", f"Precision@{limit}", "MRR", f"NDCG@{limit}", "耗时"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    best = {"recall": ("", 0), "precision": ("", 0), "mrr": ("", 0), "ndcg": ("", 0)}
    recall_key = f"recall@{limit}"
    prec_key = f"precision@{limit}"
    ndcg_key = f"ndcg@{limit}"

    for row in results["summary_table"]:
        r, p, m, n = row[recall_key], row[prec_key], row["mrr"], row[ndcg_key]
        d = row["duration_sec"]
        lines.append(f"| {row['variant']} | {r:.3f} | {p:.3f} | {m:.3f} | {n:.3f} | {d:.1f}s |")
        if r > best["recall"][1]:
            best["recall"] = (row["variant"], r)
        if p > best["precision"][1]:
            best["precision"] = (row["variant"], p)
        if m > best["mrr"][1]:
            best["mrr"] = (row["variant"], m)
        if n > best["ndcg"][1]:
            best["ndcg"] = (row["variant"], n)

    lines.append(f"")
    lines.append(f"### 各指标最佳")
    lines.append(f"- Recall@{limit}: **{best['recall'][0]}** ({best['recall'][1]:.3f})")
    lines.append(f"- Precision@{limit}: **{best['precision'][0]}** ({best['precision'][1]:.3f})")
    lines.append(f"- MRR: **{best['mrr'][0]}** ({best['mrr'][1]:.3f})")
    lines.append(f"- NDCG@{limit}: **{best['ndcg'][0]}** ({best['ndcg'][1]:.3f})")
    lines.append(f"")

    # ── 提升分析 ──
    lines.append(f"## 消融分析")
    lines.append(f"")

    baseline = results["summary_table"][0]  # BM25-only
    full = results["summary_table"][-1]     # Full pipeline
    recall_gain = (full[recall_key] - baseline[recall_key]) / max(baseline[recall_key], 0.001) * 100
    mrr_gain = (full["mrr"] - baseline["mrr"]) / max(baseline["mrr"], 0.001) * 100

    lines.append(f"相对于 BM25-only 基线:")
    lines.append(f"- Recall@{limit} 提升: **{recall_gain:+.0f}%** ({baseline[recall_key]:.3f} → {full[recall_key]:.3f})")
    lines.append(f"- MRR 提升: **{mrr_gain:+.0f}%** ({baseline['mrr']:.3f} → {full['mrr']:.3f})")
    lines.append(f"")

    # 逐层贡献
    if len(results["summary_table"]) >= 4:
        steps = results["summary_table"]
        lines.append(f"### 各层边际贡献")
        lines.append(f"")

        for i in range(1, len(steps)):
            prev = steps[i - 1]
            curr = steps[i]
            delta_r = curr[recall_key] - prev[recall_key]
            delta_m = curr["mrr"] - prev["mrr"]
            label = curr["variant"].split("(")[0].strip()
            sign_r = "+" if delta_r >= 0 else ""
            sign_m = "+" if delta_m >= 0 else ""
            lines.append(f"- **{label}**: Recall {sign_r}{delta_r:.3f}, MRR {sign_m}{delta_m:.3f}")

    lines.append(f"")

    # ── 按难度分解 ──
    lines.append(f"## 按难度分解")
    lines.append(f"")
    for difficulty in ["easy", "medium", "hard", "trap"]:
        queries = [r for r in results["per_query"]
                   if r["difficulty"] == difficulty and r["variant"] == "full_pipeline"]
        if not queries:
            continue
        avg_recall = sum(r[recall_key] for r in queries if r["relevant_ids"]) / max(len([r for r in queries if r["relevant_ids"]]), 1)
        avg_mrr = sum(r["mrr"] for r in queries if r["relevant_ids"]) / max(len([r for r in queries if r["relevant_ids"]]), 1)
        q_list = ", ".join(r["query_id"] for r in queries)
        lines.append(f"- **{difficulty}** ({q_list}): Recall@{limit}={avg_recall:.3f}, MRR={avg_mrr:.3f}")

    lines.append(f"")

    # ── 逐查询详情 ──
    lines.append(f"## 逐查询详情（完整方案）")
    lines.append(f"")
    lines.append(f"| ID | Query | 预期论文 | 实际检索 | R@{limit} | MRR |")
    lines.append(f"|----|-------|----------|----------|------|-----|")

    full_per_query = [
        r for r in results["per_query"]
        if r["variant"] == "full_pipeline"
    ]
    # 按 query_id 排序
    full_per_query.sort(key=lambda r: r["query_id"])
    for r in full_per_query:
        expected = r["relevant_ids"]
        actual = r["retrieved_ids"]
        expected_str = str(expected) if expected else "（无，负样本测试）"
        actual_str = str(actual)
        rk = r[recall_key]
        mk = r["mrr"]
        lines.append(f"| {r['query_id']} | {r['query'][:30]}... | {expected_str} | {actual_str} | {rk:.2f} | {mk:.2f} |")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*报告生成于 eval 用户知识库 ({results['paper_count']} 篇论文)*")

    return "\n".join(lines)
