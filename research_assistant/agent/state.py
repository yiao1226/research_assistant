"""LangGraph 状态定义 — 科研工作流的共享状态。

整个工作流中所有节点共享同一个 ResearchState，
每个节点读取需要的字段，写入处理结果。
LangGraph Checkpointer 在每个节点完成后自动持久化状态。

非可序列化字段（_storage, _kb_search_func 等）通过 RuntimeContext
单独传递，避免 SqliteSaver 序列化失败。
"""
from __future__ import annotations

import threading
from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages


class PaperInfo(TypedDict, total=False):
    """单篇论文的结构化信息。"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    doi: str
    categories: list[str]
    citation_count: int          # Semantic Scholar 引用数
    influential_citation_count: int
    relevance_score: float       # 与主题的相关性评分 (0-10)
    source: str                  # 来源: "arxiv" / "semantic_scholar" / "pubmed"


class AnalyzedPaper(TypedDict, total=False):
    """深度分析后的论文。"""
    paper: PaperInfo
    core_contribution: str       # 核心贡献
    method_summary: str           # 方法概述
    key_findings: str             # 关键发现
    strengths: list[str]          # 优势
    limitations: list[str]        # 局限
    relevance_to_my_research: str # 与我的研究关联
    suggested_followups: list[str] # 建议追踪的后续工作


class UserProgressEntry(TypedDict, total=False):
    """用户自己记录的实验/研究进展。"""
    timestamp: str
    entry_type: str               # "experiment" / "reading" / "idea" / "result" / "other"
    title: str
    content: str
    results: Optional[str]        # 实验结果（数值/图表描述）
    insights: Optional[str]       # 从中获得的洞察
    next_actions: Optional[str]   # 接下来的行动


class ResearchState(TypedDict, total=False):
    """科研工作流完整状态 — Agent 驱动版（4节点）。

    LangGraph 在每个节点执行后自动通过 Checkpointer 持久化此状态。
    所有字段必须是 JSON 可序列化的。
    """

    # === 用户输入 ===
    topic: str                               # 研究主题
    workflow_type: str                       # "review" / "research" / "progress_report"
    research_goal: str                       # 研究目标（向后兼容）
    additional_requirements: str             # 额外要求

    # === 消息历史（LangGraph 标准字段） ===
    messages: Annotated[list, add_messages]   # 对话历史，自动追加

    # === 搜索阶段（新） ===
    search_plan: str                         # understand 节点产出的搜索策略
    papers_found: list[dict]                  # 搜索到的论文列表（跨轮累积）
    search_iterations: int                   # 当前搜索轮次
    agent_satisfied: bool                    # Agent 是否满意搜索结果
    search_summary: str                      # Agent 搜索评估摘要
    search_queries_used: list[str]            # 向后兼容
    search_status: str                        # 向后兼容

    # === 论文分析（向后兼容） ===
    analyzed_papers: list[dict]
    papers_skipped: list[str]

    # === 综合输出（新） ===
    final_output: str                        # 最终产出（综述/回答/报告）
    cited_papers: list[dict]                 # 引用的论文列表
    literature_review: str                   # 向后兼容
    review_word_count: int                   # 向后兼容

    # === 人机协同（新） ===
    user_feedback: str                       # user_review 节点输入
    output_approved: bool                    # 用户确认通过

    # === 研究计划 ===
    research_plan: str
    suggested_experiments: list[str]

    # === 进展记录 ===
    user_progress_entries: list[dict]
    pending_user_input: Optional[str]
    progress_report: str
    knowledge_gaps: list[str]
    next_step_suggestions: list[str]

    # === 元数据 ===
    current_stage: str                       # 当前工作流阶段
    error_message: Optional[str]             # 错误信息


# ============================================================
# Runtime Context — 不可序列化的运行时依赖
# ============================================================

_runtime_context: dict[str, dict] = {}
_runtime_lock = threading.Lock()


def set_runtime_context(thread_id: str, **kwargs):
    """设置工作流的运行时上下文（不可序列化字段）。

    在 graph.stream() 之前调用，将 _storage, _kb_search_func 等
    无法被 SqliteSaver 序列化的对象存入线程安全的上下文字典。

    Args:
        thread_id: LangGraph thread_id，用于隔离不同工作流实例
        **kwargs: 要存储的上下文键值对
    """
    with _runtime_lock:
        _runtime_context[thread_id] = kwargs


MAX_RUNTIME_CONTEXTS = 100  # 防止内存泄漏

def get_runtime_context(thread_id: str | None = None) -> dict:
    """获取指定工作流的运行时上下文。

    Args:
        thread_id: LangGraph thread_id。如果为 None，返回第一个匹配的上下文
                   （仅向后兼容单用户 CLI 场景）。

    Returns:
        上下文字典，未设置时返回空字典
    """
    with _runtime_lock:
        if thread_id:
            ctx = _runtime_context.get(thread_id)
            return dict(ctx) if ctx else {}
        # 无 thread_id 时降级到遍历（仅单用户场景）
        for ctx in _runtime_context.values():
            return dict(ctx)
    return {}


def clear_runtime_context(thread_id: str = None):
    """清空运行时上下文。

    Args:
        thread_id: 指定 thread_id 则只清该实例，否则全清
    """
    with _runtime_lock:
        if thread_id:
            _runtime_context.pop(thread_id, None)
        else:
            _runtime_context.clear()
        # 防止内存泄漏：超过上限时删除最旧的条目
        excess = len(_runtime_context) - MAX_RUNTIME_CONTEXTS
        if excess > 0:
            oldest = sorted(_runtime_context.keys())[:excess]
            for k in oldest:
                _runtime_context.pop(k, None)
