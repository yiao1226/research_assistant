"""评估编排器 — 遍历检索变体 × 评估查询 → 计算指标 → 产出结构化结果。

用法:
  python -m research_assistant.eval.runner

或:
  from research_assistant.eval import run_eval
  results = run_eval()
"""

from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
from .queries import EVAL_QUERIES, VARIANTS

logger = logging.getLogger(__name__)


def _setup_storage():
    """初始化 eval 用户的存储。"""
    from research_assistant.core.storage import PerUserStorage
    user_dir = Path("./data/users/eval")
    storage = PerUserStorage(user_dir)
    return storage


def _setup_retriever(storage):
    """初始化混合检索器。"""
    from research_assistant.rag.retrieval import HybridRetriever
    return HybridRetriever(storage, "eval")


def _run_variant_bm25_only(retriever, query: str, limit: int) -> list[int]:
    """纯 BM25 检索。"""
    results = retriever.bm25_search(query, limit=limit)
    return [int(r.get("paper_id", r.get("id", 0))) for r in results if r.get("paper_id") or r.get("id")]


def _run_variant_dense_only(retriever, query: str, limit: int) -> list[int]:
    """纯 Dense 向量检索。"""
    results = retriever.dense_search("papers", query, limit=limit)
    return [int(r.get("paper_id", r.get("id", 0))) for r in results if r.get("paper_id") or r.get("id")]


def _run_variant_bm25_dense_rrf(retriever, query: str, limit: int) -> list[int]:
    """BM25 + Dense + RRF 三路融合。"""
    results = retriever.search_papers(query, limit=limit)
    return [int(r.get("paper_id", r.get("id", 0))) for r in results if r.get("paper_id") or r.get("id")]


def _run_variant_full(retriever, query: str, limit: int) -> list[int]:
    """五路融合: 三路RRF + HyDE + MQE + heading_path 加权。"""
    results = retriever.search_papers_expanded(
        query, limit=limit, enable_mqe=True, mqe_expansions=3, enable_hyde=True,
    )
    return [int(r.get("paper_id", r.get("id", 0))) for r in results if r.get("paper_id") or r.get("id")]


VARIANT_FUNCTIONS = {
    "bm25_only": _run_variant_bm25_only,
    "dense_only": _run_variant_dense_only,
    "bm25_dense_rrf": _run_variant_bm25_dense_rrf,
    "full_pipeline": _run_variant_full,
}


def run_eval(limit: int = 5, verbose: bool = True) -> dict[str, Any]:
    """运行完整评估。

    Args:
        limit: 检索结果截断数量 (k)
        verbose: 是否打印进度

    Returns:
        {
            "variants": {"bm25_only": {...}, ...},
            "per_query": [...],
            "summary_table": [...],
        }
    """
    # 加载 .env（确保 QDRANT_URL 等环境变量可用）
    from dotenv import load_dotenv
    load_dotenv()

    storage = _setup_storage()
    paper_count = storage.count_papers()
    if paper_count == 0:
        return {"error": "eval 用户知识库为空，请先入库论文"}

    if verbose:
        print(f"📊 Eval: {paper_count} 篇论文, {len(EVAL_QUERIES)} 条查询, k={limit}")
        print(f"   变体数: {len(VARIANTS)}")

    retriever = _setup_retriever(storage)

    all_results = {}  # variant_name → list of (query_id, paper_ids)
    durations = {}

    for var_key, var_info in VARIANTS.items():
        if verbose:
            print(f"\n  ⏳ {var_info['label']} ...", end=" ", flush=True)

        fn = VARIANT_FUNCTIONS[var_key]
        t0 = time.time()

        variant_results = []
        for q in EVAL_QUERIES:
            try:
                ids = fn(retriever, q["query"], limit=limit)
                # 去重保序
                seen = set()
                deduped = []
                for pid in ids:
                    if pid not in seen and pid > 0:
                        seen.add(pid)
                        deduped.append(pid)
                variant_results.append((q["id"], deduped))
            except Exception as e:
                logger.warning("查询 %s 失败: %s", q["id"], e)
                variant_results.append((q["id"], []))

        elapsed = time.time() - t0
        all_results[var_key] = variant_results
        durations[var_key] = elapsed
        if verbose:
            print(f"✓ ({elapsed:.1f}s)")

    # 计算每个变体 × 每条查询的指标
    per_query_rows = []
    variant_metrics = {}  # var → aggregated metrics

    for var_key, var_info in VARIANTS.items():
        variant_results = all_results[var_key]
        total_recall = 0.0
        total_precision = 0.0
        total_mrr_score = 0.0
        total_ndcg = 0.0
        n = 0

        for q, (qid, ids) in zip(EVAL_QUERIES, variant_results):
            rel = q["relevant_ids"]
            r = recall_at_k(ids, rel, limit)
            p = precision_at_k(ids, rel, limit)
            m = mrr(ids, rel)
            nd = ndcg_at_k(ids, rel, limit)

            per_query_rows.append({
                "query_id": qid,
                "query": q["query"],
                "variant": var_key,
                "difficulty": q["difficulty"],
                "category": q["category"],
                "retrieved_ids": ids[:limit],
                "relevant_ids": sorted(rel),
                f"recall@{limit}": round(r, 3),
                f"precision@{limit}": round(p, 3),
                "mrr": round(m, 3),
                f"ndcg@{limit}": round(nd, 3),
            })

            if rel:  # 排除 negative query（没有相关文档的）
                total_recall += r
                total_precision += p
                total_mrr_score += m
                total_ndcg += nd
                n += 1

        if n > 0:
            variant_metrics[var_key] = {
                "label": var_info["label"],
                "desc": var_info["desc"],
                f"recall@{limit}": round(total_recall / n, 3),
                f"precision@{limit}": round(total_precision / n, 3),
                "mrr": round(total_mrr_score / n, 3),
                f"ndcg@{limit}": round(total_ndcg / n, 3),
                "duration_sec": round(durations[var_key], 1),
                "queries_with_relevant": n,
                        }
        else:
            variant_metrics[var_key] = {
                "label": var_info["label"],
                "desc": var_info["desc"],
                f"recall@{limit}": 0.0,
                f"precision@{limit}": 0.0,
                "mrr": 0.0,
                f"ndcg@{limit}": 0.0,
                "duration_sec": round(durations[var_key], 1),
                "queries_with_relevant": 0,
            }

    # 汇总表（用于 Markdown 报告）
    summary_table = []
    metric_keys = [f"recall@{limit}", f"precision@{limit}", "mrr", f"ndcg@{limit}"]
    for var_key in ["bm25_only", "dense_only", "bm25_dense_rrf", "full_pipeline"]:
        if var_key in variant_metrics:
            row = {"variant": variant_metrics[var_key]["label"]}
            row.update({k: variant_metrics[var_key][k] for k in metric_keys})
            row["duration_sec"] = variant_metrics[var_key]["duration_sec"]
            summary_table.append(row)

    return {
        "limit": limit,
        "paper_count": paper_count,
        "query_count": len(EVAL_QUERIES),
        "variants": variant_metrics,
        "per_query": per_query_rows,
        "summary_table": summary_table,
    }


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)

    from .report import format_eval_report
    results = run_eval(verbose=True)
    if "error" in results:
        print(f"\n❌ {results['error']}")
    else:
        print(format_eval_report(results))
