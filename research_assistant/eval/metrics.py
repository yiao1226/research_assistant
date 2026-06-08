"""检索评估指标 — 纯函数，零外部依赖。

面试可讲:
- recall@k: 相关文档被找到的比例（漏掉了多少）
- precision@k: 前k结果中相关的比例（有多少噪音）
- MRR: 第一个相关结果的排名倒数（用户要翻多久）
- NDCG@k: 考虑排序位置的归一化增益（排名质量）
"""

from __future__ import annotations
import math


def recall_at_k(retrieved: list[int], relevant: set[int], k: int = 5) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|"""
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[int], relevant: set[int], k: int = 5) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k"""
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    """MRR = 1 / rank_of_first_relevant. 0 if none found."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int = 5) -> float:
    """NDCG@k: 排序位置敏感的指标。排第1位比排第5位的贡献更大。"""
    if not relevant or k <= 0:
        return 0.0

    def _dcg(ids):
        return sum(1.0 / math.log2(i + 2) for i, d in enumerate(ids[:k]) if d in relevant)

    dcg = _dcg(retrieved)
    ideal = _dcg(sorted(relevant, key=lambda x: x))  # 所有相关排最前
    return dcg / ideal if ideal > 0 else 0.0
