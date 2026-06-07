"""科研助手 — 工具集。"""
# 轻量级：无 langchain_openai 依赖
from .paper_search import search_arxiv, search_semantic_scholar, search_web_of_science, download_paper, fetch_paper_full_text

# 重量级：延迟导入（import 时触发 langchain_openai ~7s）
_search_orchestrator = None
_upload_manager = None
_record_user_progress = None
_get_progress_summary = None
_detect_new_directions = None
_update_plan_from_progress = None


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
