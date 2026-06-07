"""科研助手 — 工具集。

工具分层:
  轻量工具（Agent/LangGraph 共用）:
    make_all_agent_tools() — 组装所有 Agent 工具的工厂函数

  API 工具:
    search_arxiv, search_semantic_scholar, search_web_of_science

  业务模块（延迟加载）:
    SearchOrchestrator, UploadManager, progress, plan
"""
from __future__ import annotations

# 轻量级 API 调用（无重量依赖）
from .paper_search import (
    search_arxiv, search_semantic_scholar, search_web_of_science,
    download_paper, fetch_paper_full_text,
)

# Agent 工具工厂
from .kb import make_kb_tools
from .search import make_search_tool
from .history import make_history_tool


def make_all_agent_tools(storage, username: str) -> list:
    """生成 Agent 的完整工具集。

    被 qa.py 和 graph.py 共用。4 个工具:
      - query_knowledge_base: 本地论文库语义检索
      - query_progress: 用户进展记录查询
      - recall_history: 历史会话回忆
      - search_papers_online: 外部论文搜索

    Args:
        storage: PerUserStorage 实例
        username: 当前用户名

    Returns:
        LangChain @tool 列表
    """
    kb_tools = make_kb_tools(storage, username)
    search_tools = make_search_tool(storage, username)
    history_tool = make_history_tool(storage, username)
    return [*kb_tools, *search_tools, history_tool]


# 重量级模块：延迟导入（触发 langchain_openai ~7s）
def __getattr__(name):
    import importlib
    _lazy_map = {
        "SearchOrchestrator": ".search_orchestrator",
        "UploadManager": ".upload",
        "record_user_progress": ".progress",
        "get_progress_summary": ".progress",
        "detect_new_directions": ".plan",
        "update_plan_from_progress": ".plan",
    }
    if name in _lazy_map:
        mod = importlib.import_module(_lazy_map[name], __package__)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
