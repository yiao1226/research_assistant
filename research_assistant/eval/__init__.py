"""检索质量评估模块 — recall@k, precision@k, MRR, NDCG + 消融实验编排。"""
from .metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
from .runner import run_eval
from .report import format_eval_report

__all__ = [
    "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
    "run_eval", "format_eval_report",
]
