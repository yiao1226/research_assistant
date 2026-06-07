"""CLI 模块 — 交互界面和命令执行。"""
from .commands import (
    get_storage, get_orchestrator, get_uploader,
    get_episodic, get_backup, get_qa,
    run_search, run_upload, run_download,
    run_progress, run_record, run_recall,
    run_qa, run_backup, run_user_cmd, run_paper_cmd, run_review,
    _on_quit,
)
from .display import print_banner
