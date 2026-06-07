#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科研助手 — 多用户持久记忆 + RAG 语义检索 + 智能问答 + 个性化文献分析。

用法:
  python main.py                          — 交互模式
  python main.py ask <问题>               — 智能问答
  python main.py search [-s 源] 关键词    — 论文搜索
  python main.py upload <pdf_path>        — 上传论文
  python main.py review <主题>            — 文献综述
  python main.py progress <主题>          — 查看进展
  python main.py recall <问句>            — 跨会话回忆
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# 屏蔽第三方库的弃用警告（jieba -> pkg_resources）
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s [%(name)s] %(message)s')

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_assistant.core import UserManager
from research_assistant.tools import search_arxiv, search_semantic_scholar, search_web_of_science

# VectorStore 延迟导入：BGE 模型 96MB，登录后再加载
VectorStore = None

from cli.display import print_banner, HELP_TEXT, COMMAND_LIST
from cli.commands import (
    get_storage, get_episodic, get_backup, get_qa,
    run_search, run_upload, run_download, run_progress,
    run_record, run_recall, run_qa, run_backup,
    run_user_cmd, run_paper_cmd, run_review, _on_quit,
)
import cli.commands as cmd_mod

# ── 全局状态 ──

BASE_DIR = Path("./data")
user_manager = UserManager(".")

AVAILABLE_SOURCES = {
    "arxiv": ("ArXiv", search_arxiv),
    "s2": ("Semantic Scholar", search_semantic_scholar),
    "wos": ("Web of Science", search_web_of_science),
}
TYPE_LABELS = {
    "experiment": "实验", "reading": "文献阅读",
    "idea": "新想法", "result": "阶段性结果",
    "meeting": "讨论会", "other": "其他",
}

# 注入全局引用到 commands 模块
cmd_mod.user_manager = user_manager
cmd_mod.BASE_DIR = BASE_DIR
cmd_mod.AVAILABLE_SOURCES = AVAILABLE_SOURCES
cmd_mod.TYPE_LABELS = TYPE_LABELS


# ── 交互模式 ──

def interactive_mode():
    print_banner()
    users = user_manager.list_users()
    if users:
        print(f"\n现有用户: {', '.join(users)}")
        name = input("请输入用户名: ").strip()
    else:
        print("\n欢迎首次使用！")
        name = input("请创建用户名: ").strip()
    if not name:
        name = "default"

    if not user_manager.user_exists(name):
        confirm = input(f"用户 '{name}' 不存在，确认创建新用户？(Y/n): ").strip().lower()
        if confirm and confirm != 'y':
            print("已取消。")
            return

    user = user_manager.login(name)
    user.init_storage()

    # MarkItDown 预检
    try:
        import markitdown  # noqa: F401
        print(f"  [OK] MarkItDown PDF 解析就绪")
    except ImportError:
        print(f"\n  [INFO] MarkItDown 未安装，PDF 解析将使用 PyMuPDF 兜底。")

    # Qdrant collections（延迟导入：BGE 模型 96MB，等用户登录后再加载）
    global VectorStore
    if VectorStore is None:
        from research_assistant.rag import VectorStore as VS
        VectorStore = VS
    vs = VectorStore()
    vs.ensure_all_collections(name)

    # 完整性检查
    backup = get_backup()
    if backup:
        integrity = backup.verify_integrity()
        if integrity.get("issues"):
            print(f"\n[INTEGRITY] {len(integrity['issues'])} 个问题:")
            for issue in integrity["issues"]:
                print(f"  - {issue}")

    ctx = user_manager.get_last_session_context()
    print(f"\n当前用户: {name}")
    if ctx and ctx.get("last_login"):
        print(f"上次登录: {ctx.get('last_login', '')[:19]}")
    print(f"已保存论文: {user.paper_count()} 篇 | 进展记录: {user.progress_count()} 条")
    if ctx and ctx.get("recent_searches"):
        print(f"最近搜索: {' / '.join(ctx['recent_searches'][:3])}")

    # 未解决问题
    ep = get_episodic()
    if ep:
        recent = ep.get_recent_sessions(limit=1)
        if recent:
            details = recent[0].get("details", {})
            if isinstance(details, str):
                try: details = json.loads(details)
                except Exception: details = {}
            unresolved = details.get("unresolved_questions", [])
            if unresolved:
                print("\n上次未解决问题:")
                for q in unresolved[:3]:
                    print(f"  - {q}")

    print(COMMAND_LIST)

    # 命令分发
    dispatch = {
        "help": lambda _: print(HELP_TEXT),
        "search": run_search,
        "upload": run_upload,
        "download": run_download,
        "review": run_review,
        "progress": run_progress,
        "record": run_record,
        "recall": run_recall,
        "ask": run_qa,
        "backup": lambda _: run_backup(),
        "user": run_user_cmd,
        "paper": run_paper_cmd,
    }

    while True:
        try:
            cmd = input(f"\n[LAB] {name} > ").strip()
        except (EOFError, KeyboardInterrupt):
            _on_quit()
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if action in ("quit", "exit", "q"):
            _on_quit()
            break

        fn = dispatch.get(action)
        if fn:
            try:
                fn(arg) if arg else fn()
            except TypeError:
                fn(arg)
        else:
            print(f"未知命令: {action}。输入 help 查看可用命令。")


# ── 命令行模式 ──

CMDLINE_CMDS = {
    "review": lambda arg: run_review(topic=arg),
    "search": lambda arg: run_search(arg=arg),
    "upload": lambda arg: run_upload(arg=arg),
    "download": lambda arg: run_download(arg=arg),
    "progress": lambda arg: run_progress(topic=arg),
    "recall": lambda arg: run_recall(arg=arg),
    "ask": lambda arg: run_qa(arg=arg),
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        user_manager.login("default")
        get_storage()
        fn = CMDLINE_CMDS.get(action)
        if fn and arg:
            fn(arg)
        else:
            print("用法: python main.py [ask|search|upload|review|download|progress|recall] <参数>")
            print("      python main.py  (交互模式)")
    else:
        interactive_mode()
