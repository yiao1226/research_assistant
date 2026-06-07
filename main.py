#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科研助手 — 多用户持久记忆 + RAG 语义检索 + 智能问答 + 个性化文献分析。

用法:
  python main.py                          — 交互模式（默认 Agent 智能路由）
  python main.py "金刚石CVD温度优化"       — 单次 Agent 问答
  python main.py /search 关键词            — 快速命令模式

交互模式:
  直接输入问题 → Agent 自动分析 + 工具调用
  /command     → 快速命令通道
  拖入文件路径  → 自动识别 + 分析 + 入库建议
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
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
    run_user_cmd, run_paper_cmd, run_review,
    run_research, run_progress_report, _on_quit,
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


# ── 文件路径检测 ──

# 已知文档扩展名（来自 DocumentLoader）
_DOC_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
}

_FILE_PATH_RE = re.compile(
    r'^([A-Za-z]:[\\/]|\.\.?[\\/]|~?[\\/])'  # C:\ 或 ../ 或 .\ 或 ~/
)

def _is_file_path(text: str) -> bool:
    """检测输入是否为文件路径（不调 os.path.exists，避免网络延迟）。"""
    if not text:
        return False

    # 快速排除：纯自然语言
    stripped = text.strip().strip('"').strip("'")

    # 1. 正则匹配路径模式
    if _FILE_PATH_RE.match(stripped):
        ext = os.path.splitext(stripped)[1].lower()
        if ext in _DOC_EXTENSIONS:
            return True
        # 无扩展名但路径格式明显
        if ext == '' and len(stripped) > 10:
            return os.path.exists(stripped)

    # 2. 全路径可能在引号内
    if os.path.exists(stripped) and os.path.isfile(stripped):
        return True

    return False


def _handle_file_input(path: str):
    """处理用户拖入的文件路径: 加载 → 分析 → 交互选择。"""
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    ext = os.path.splitext(path)[1].lower()
    if ext not in _DOC_EXTENSIONS:
        print(f"不支持的文件类型: {ext}。支持: {', '.join(sorted(_DOC_EXTENSIONS))}")
        return

    print(f"\n📄 正在分析: {os.path.basename(path)}")

    try:
        from research_assistant.loaders import load_document
        from research_assistant.utils import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        # 加载文档
        full_text, metadata = load_document(path)
        print(f"   解析完成: {metadata.get('pages', '?')}页, "
              f"{len(full_text)}字符 ({metadata.get('extraction_method', '?')})")

        # LLM 快速分析
        llm = get_llm(temperature=0.2, max_tokens=512)
        title = metadata.get('title', '未知')
        abstract = metadata.get('abstract', '')
        analysis_prompt = f"""分析以下论文信息，给出简洁判断：

标题: {title}
摘要: {(abstract or full_text)[:800]}
页数: {metadata.get('pages', '?')}

请用1-2句话说明:
1. 这篇论文的研究主题和方法
2. 建议: 是否值得入库（考虑: 学术价值、方法详实度、页数合理性）

直接回答，不要JSON。"""

        response = llm.invoke([
            SystemMessage(content="你是学术论文分析助手。简洁直接。"),
            HumanMessage(content=analysis_prompt),
        ])
        analysis = str(response.content).strip()

        print(f"\n{'─' * 55}")
        print(f"  标题: {title[:70]}")
        if metadata.get('authors'):
            authors = metadata['authors']
            if isinstance(authors, list):
                print(f"  作者: {', '.join(authors[:3])}")
        print(f"  页数: {metadata.get('pages', '?')}")
        print(f"  分析: {analysis}")
        print(f"{'─' * 55}")

        # 交互选择
        print("\n  [I] 入库到知识库  [Q] 追问内容  [S] 跳过")
        choice = input("  > ").strip().lower()

        if choice == 'i':
            uploader = cmd_mod.get_uploader()
            if uploader:
                result = uploader.upload_file(path, confirm=False)
                print(f"  {result.get('message', '')}")
            else:
                print("  请先登录用户。")
        elif choice == 'q':
            question = input("  想问什么？> ").strip()
            if question:
                qa = get_qa()
                if qa:
                    result = qa.ask(
                        f"关于刚加载的论文《{title}》(摘要: {(abstract or full_text)[:500]}), "
                        f"用户问: {question}"
                    )
                    print(f"\n{result['answer']}")
        # 's' 或其他 → 跳过

    except Exception as e:
        logger.warning("文件分析失败", exc_info=True)
        print(f"  分析失败: {e}")


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

    # 斜杠命令分发（快速通道）
    dispatch = {
        "search": run_search, "s": run_search,
        "upload": run_upload, "u": run_upload,
        "download": run_download, "d": run_download,
        "review": run_review, "r": run_review,
        "research": run_research, "rs": run_research,
        "progress-report": run_progress_report, "pr": run_progress_report,
        "progress": run_progress, "p": run_progress,
        "record": run_record, "n": run_record,
        "recall": run_recall,
        "ask": run_qa, "a": run_qa,
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

        # ── 裸命令（无需 / 前缀）──
        if cmd.lower() in ("quit", "exit", "q"):
            _on_quit()
            break

        if cmd.lower() == "help":
            print(HELP_TEXT)
            continue

        # ── 斜杠命令（快速通道）──
        if cmd.startswith("/"):
            parts = cmd[1:].split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            fn = dispatch.get(action)
            if fn:
                try:
                    fn(arg) if arg else fn()
                except TypeError:
                    fn(arg)
            else:
                print(f"未知命令: /{action}。输入 help 查看可用命令。")
            continue

        # ── 文件路径检测（拖入 PDF）──
        if _is_file_path(cmd):
            _handle_file_input(cmd)
            continue

        # ── 默认：Agent 智能问答 ──
        qa = get_qa()
        if not qa:
            print("请先登录用户。")
            continue

        print()
        result = qa.ask(cmd)

        # 工具调用日志
        tool_calls = result.get('tool_calls', [])
        if tool_calls:
            tools_used = {tc['tool'] for tc in tool_calls}
            print(f'🔧 调用了: {", ".join(tools_used)}  ({len(tool_calls)}次)')

        # 流式输出已在 agent_loop 里打印过，不重复
        if not result.get('streamed'):
            print(f'\n{"=" * 60}')
            print(result['answer'])
            print(f'{"=" * 60}')

        cited = result.get('cited_papers', [])
        if cited:
            print(f'\n📚 引用论文: {len(cited)}篇')
            for i, p in enumerate(cited[:5], 1):
                print(f'  [{i}] {p.get("title","?")[:60]}')


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

        # 裸参数 → Agent 模式 (python main.py "金刚石CVD温度优化")
        if action == arg and not action.startswith("/"):
            action = "ask"
            arg = " ".join(sys.argv[1:])
        # /command 格式
        elif action.startswith("/"):
            action = action[1:]
            arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

        user_manager.login("default")
        get_storage()
        fn = CMDLINE_CMDS.get(action)
        if fn and arg:
            fn(arg)
        elif action == "ask":
            qa = get_qa()
            if qa:
                result = qa.ask(arg)
                if not result.get("streamed"):
                    print(result["answer"])
        else:
            # 降级: 全部当 Agent 输入
            qa = get_qa()
            if qa:
                full = " ".join(sys.argv[1:])
                result = qa.ask(full)
                if not result.get("streamed"):
                    print(result["answer"])
    else:
        interactive_mode()
