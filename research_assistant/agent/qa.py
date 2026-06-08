"""智能问答服务 — Agent 循环 + Hook + 动态工具映射。

核心设计（参考 Claude Code 架构）:
  agent_loop() → 可见的循环体，Hook 挂横切逻辑
  build_system() → 工具描述从定义生成，不硬编码
  hooks → PreToolUse / PostToolUse / BeforeLLM / AfterLLM
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..utils import get_llm
from ..core.storage import PerUserStorage
from ..rag.retrieval import HybridRetriever
from ..memory.working import WorkingMemory
from ..tools import make_all_agent_tools

logger = logging.getLogger(__name__)

# ============================================================
# Hook 系统
# ============================================================

HOOKS: dict[str, list[Callable]] = {
    "before_llm": [],      # (messages) → None
    "after_llm": [],       # (response) → None
    "pre_tool": [],        # (tool_name, tool_args) → str|None (返回非None=拦截)
    "post_tool": [],       # (tool_name, tool_args, result) → None
    "agent_start": [],     # (question) → None
    "agent_end": [],       # (result) → None
}


def register_hook(event: str, callback: Callable):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """触发 Hook，pre_tool 的拦截值会传播。"""
    for cb in HOOKS[event]:
        result = cb(*args)
        if event == "pre_tool" and result is not None:
            return result  # 拦截
    return None


# ═══════════════════════════════════════════════════════════
# 内置 Hook: 进度显示 + 工具日志
# ═══════════════════════════════════════════════════════════

def _progress_hook(question: str):
    """agent_start: 显示当前轮次和问题。"""
    import sys
    print(f"\n🤖 分析: {question[:60]}...", flush=True)

def _tool_log_hook(name: str, args: dict):
    """pre_tool: 显示工具调用。"""
    import sys
    arg_preview = {k: str(v)[:60] for k, v in args.items()}
    label = {"search_papers_online": "🔍 外部搜索", "query_knowledge_base": "📚 知识库检索",
             "query_progress": "📋 进展查询", "recall_history": "🧠 历史回忆"}.get(name, "🔧")
    print(f"  {label} {name}({arg_preview})", flush=True)
    return None

def _tool_result_hook(_name: str, _args: dict, result: str):
    """post_tool: 显示工具结果摘要。"""
    import sys
    preview = str(result)[:120].replace("\n", " ")
    print(f"     → {preview}", flush=True)

def _agent_done_hook(result: dict):
    """agent_end: 显示统计。"""
    calls = len(result.get("tool_calls", []))
    if calls:
        tools_used = {tc["tool"] for tc in result["tool_calls"]}
        print(f"✅ 完成 ({calls}次工具调用: {', '.join(tools_used)})")


register_hook("agent_start", _progress_hook)
register_hook("pre_tool", _tool_log_hook)
register_hook("post_tool", _tool_result_hook)
register_hook("agent_end", _agent_done_hook)


# ═══════════════════════════════════════════════════════════
# 内置 Hook: Qdrant 离线降级兜底
# ═══════════════════════════════════════════════════════════

# 依赖 Qdrant 的工具映射
_QDRANT_TOOLS = {
    "query_knowledge_base": "知识库检索",
    "query_progress": "进展查询",
    "recall_history": "历史回忆",
    "ingest_papers": "论文入库",
}

# 健康检查缓存：避免每次工具调用都探测
_qdrant_health = {"alive": True, "checked_at": 0.0}
_QDRANT_CHECK_TTL = 30.0  # 30 秒内复用缓存结果


def _probe_qdrant() -> bool:
    """探测 Qdrant 是否在线。

    30 秒内缓存结果，避免每次工具调用增加 ~100ms 网络开销。
    检测用轻量操作（get_collections），超时 2 秒。
    """
    now = time.time()
    if now - _qdrant_health["checked_at"] < _QDRANT_CHECK_TTL:
        return _qdrant_health["alive"]

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            timeout=2.0,
        )
        client.get_collections()
    except Exception:
        _qdrant_health["alive"] = False
    else:
        _qdrant_health["alive"] = True

    _qdrant_health["checked_at"] = time.time()
    return _qdrant_health["alive"]


def _qdrant_fallback_hook(name: str, _args: dict):
    """pre_tool: Qdrant 离线时拦截依赖工具，返回降级提示。

    只拦截依赖 Qdrant 的工具（搜索/入库）。
    search_papers_online 走外网 API，不受影响，不拦截。
    返回非 None → 工具真实逻辑被跳过，LLM 收到降级文本。
    """
    if name not in _QDRANT_TOOLS:
        return None  # 不受影响的工具（如 search_papers_online）

    if _probe_qdrant():
        return None  # Qdrant 在线，放行

    label = _QDRANT_TOOLS[name]

    if name == "ingest_papers":
        return (
            f"（{label}暂时不可用——向量数据库离线。"
            "论文元数据已保存到 SQLite，向量索引将在恢复后重建。"
            "请告知用户：启动 Docker Qdrant 容器后重试入库。）"
        )
    else:
        return (
            f"（{label}暂时不可用——向量数据库离线。"
            "请基于你的训练知识和对话历史回答用户问题，"
            "并建议用户执行 docker start qdrant 启动向量数据库。）"
        )


register_hook("pre_tool", _qdrant_fallback_hook)


# ═══════════════════════════════════════════════════════════
# 内置 Hook: 后台任务通知注入 + 慢操作路由
# ═══════════════════════════════════════════════════════════

# 模块级工具引用：_bg_dispatch_hook 需要访问工具列表来做后台派发
_current_tools: list = []


def _bg_notification_hook(messages: list):
    """before_llm: 将已完成的后台任务结果注入消息列表。

    在每轮 LLM 调用前检查是否有后台任务完成，
    有则拼入 HumanMessage 让 LLM 看到。
    """
    from .background import get_bg_manager
    try:
        mgr = get_bg_manager()
    except RuntimeError:
        return  # 未初始化，跳过
    notifications = mgr.collect_notifications()
    if notifications:
        from langchain_core.messages import HumanMessage
        content = "[后台任务完成通知]\n\n" + "\n\n".join(notifications)
        messages.append(HumanMessage(content=content))
        import sys
        print(f"\n📨 后台任务结果已注入 ({len(notifications)} 条)", flush=True)


def _bg_dispatch_hook(name: str, args: dict):
    """pre_tool: 慢操作路由到后台线程。

    返回 "__BG_DISPATCHED__" sentinel 标记，
    由 agent_loop 识别并跳过 _exec_tool 调用。
    """
    from .background import _is_slow_tool, get_bg_manager
    if not _is_slow_tool(name):
        return None  # 快速工具，不拦截

    try:
        mgr = get_bg_manager()
    except RuntimeError:
        return None  # 未初始化，降级为同步执行

    if mgr.is_shutting_down:
        return None  # 关闭中，降级为同步执行

    # 找到对应的工具实例
    tool = None
    for t in _current_tools:
        if t.name == name:
            tool = t
            break
    if tool is None:
        return None

    # 构造命令描述
    arg_preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
    command = f"{name}({arg_preview})"

    import sys
    print(f"  ⏳ 后台派发 {command[:60]}...", flush=True)

    # 派发后台任务
    task_id = mgr.dispatch(
        tool_name=name,
        command=command,
        fn=lambda: str(tool.invoke(args) if tool else "（工具不可用）"),
    )

    return f"[后台任务 {task_id} 已启动]\n{command}\n结果将在完成后自动通知你。"


register_hook("before_llm", _bg_notification_hook)
register_hook("pre_tool", _bg_dispatch_hook)


# ============================================================
# System Prompt 动态组装（工具描述从定义生成，不硬编码）
# ============================================================

AGENT_SYSTEM_TEMPLATE = """你是科研助手，帮助用户进行学术研究。

## 可用工具

{tools}

## 决策原则

- 问候/闲聊/概念解释: 不调工具，直接友好回答
- 科研问题(需要数据): 先调 query_knowledge_base，不够再调其他
- 回忆历史: 调 recall_history
- 找新论文/外部搜索: 调 search_papers_online → 结果展示后会交互询问入库
- 可组合调用多个工具
- 工具无结果时诚实告知，不编造

## 重要: 搜索后不要直接写综述

- search_papers_online 搜完后，工具本身会向用户展示结果并询问入库
- 用户选择入库后，你可以简要总结论文方向
- 只有当用户明确说"写综述"/"review"时，才进入综述模式
- 普通搜索只是展示结果 + 用户选择，不需要自动综述

## 回答格式

- 基于工具返回的真实内容回答，引用论文用 [论文标题] 标注
- 包含具体数值、参数、指标
- 中文回答，学术风格

## 背景上下文

{context}"""


def build_system_prompt(tools: list, context: str) -> str:
    """从工具定义动态生成 System Prompt。

    工具描述来自 @tool 的 docstring，不用人工维护。
    """
    tool_lines = []
    for i, t in enumerate(tools, 1):
        # 从工具定义中提取 name + description
        desc = t.description.split("\n")[0] if t.description else str(t.name)
        tool_lines.append(f"{i}. **{t.name}** — {desc}")
    return AGENT_SYSTEM_TEMPLATE.format(
        tools="\n".join(tool_lines),
        context=context,
    )


# ============================================================
# Agent 循环（可见的循环体 + 流式输出）
# ============================================================

def agent_loop(
    messages: list,
    tools: list,
    llm,
    max_rounds: int = 3,
    stream: bool = True,
) -> tuple[list, list]:
    """Agent 循环 — 工具调用用 invoke，最终回答用 stream。

    设计:
      - 工具调用阶段: invoke → 需要完整 tool_calls 才能执行
      - 最终回答: stream → 逐 token 输出，用户看到打字效果

    Args:
        messages: [SystemMessage, HumanMessage, ...]
        tools: LangChain @tool 列表
        llm: bind_tools 后的 LLM（invoke 用）
        max_rounds: 最多工具调用轮次
        stream: 最终回答是否流式输出

    Returns:
        (messages, tool_calls_log)
    """
    tool_log = []

    for round_num in range(max_rounds):
        trigger_hooks("before_llm", messages)

        # ── 先用 invoke 检测是否有工具调用 ──
        response = llm.invoke(messages)
        has_tools = hasattr(response, 'tool_calls') and response.tool_calls

        # ── 有工具调用 → 执行后继续 ──
        if has_tools:
            trigger_hooks("after_llm", response)
            messages.append(response)
            for tc in response.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                tid = tc.get("id", "")

                blocked = trigger_hooks("pre_tool", name, args)
                result = str(blocked) if blocked else _exec_tool(tc, tools)
                tool_log.append({"tool": name, "args": args, "result_len": len(str(result))})
                trigger_hooks("post_tool", name, args, result)

                messages.append(ToolMessage(content=str(result), tool_call_id=tid))
            continue

        # ── 无工具调用 → 流式输出最终回答 ──
        if stream:
            _stream_answer(llm, messages, response)
        else:
            messages.append(response)

        trigger_hooks("after_llm", messages[-1])
        break

    return messages, tool_log


def _stream_answer(llm, messages, invoke_response):
    """输出最终回答 — 使用已获取的 invoke_response，模拟流式逐字输出。

    关键设计:
      - invoke_response 由 agent_loop 中的 llm.invoke() 已获取，不再额外调用 LLM
      - 本地逐字打印模拟流式效果（无需 API 调用）
      - 内容与 LLM 实际输出完全一致（来自同一次 invoke）

    容错: invoke_response 不可用时降级为空输出。
    """
    import sys

    text = ""
    try:
        text = str(invoke_response.content) if hasattr(invoke_response, 'content') else str(invoke_response)
    except Exception:
        text = ""

    if not text:
        messages.append(invoke_response)
        return

    # 本地模拟流式：逐字输出（间隔 0.008s ≈ 125 字/秒，接近 DeepSeek 流式速度）
    try:
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
    except Exception:
        # 逐字输出失败 → 一次性输出
        sys.stdout.write(text)
        sys.stdout.flush()

    messages.append(invoke_response)
    print()


# ═══════════════════════════════════════════════════════════
# QAService
# ═══════════════════════════════════════════════════════════

class QAService:
    """智能问答服务。"""

    def __init__(self, username: str, storage: PerUserStorage):
        self.username = username
        self.storage = storage
        self.retriever = HybridRetriever(storage, username)
        self.working = WorkingMemory(username, str(storage.user_dir), storage=self.storage)

    def ask(self, question: str) -> dict:
        """主入口: 组装 → agent_loop → 回答。"""
        trigger_hooks("agent_start", question)

        # 初始化后台任务管理器
        from .background import get_bg_manager
        get_bg_manager(self.storage, self.username)

        # 工具 + 动态 System Prompt
        tools = make_all_agent_tools(self.storage, self.username)
        _current_tools.clear()
        _current_tools.extend(tools)

        context = self._build_context()
        system = build_system_prompt(tools, context)

        llm = get_llm(temperature=0.3, max_tokens=2048)
        llm_with_tools = llm.bind_tools(tools)

        # 注册上下文压缩（捕获 llm 用于 L3）
        from .compaction import compaction_hook
        compaction = lambda msgs: compaction_hook(msgs, llm_with_tools)
        HOOKS["before_llm"].append(compaction)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=question),
        ]

        # ── Agent 循环 ──
        messages, tool_log = agent_loop(messages, tools, llm_with_tools)

        # 清理
        HOOKS["before_llm"].remove(compaction)
        _current_tools.clear()

        # 提取回答和引用
        final = messages[-1]
        answer = str(final.content) if hasattr(final, 'content') else ""
        cited = _extract_cited_from_messages(messages)

        self.working.add_turn(question, answer, cited, "")

        result = {"answer": answer, "cited_papers": cited, "tool_calls": tool_log, "streamed": True}
        trigger_hooks("agent_end", result)
        return result

    def _build_context(self) -> str:
        parts = []
        try:
            from ..memory.user_profile import UserProfileManager
            profile_mgr = UserProfileManager(self.username, str(self.storage.user_dir))
            text = profile_mgr.build_context_for_qa()
            if text:
                parts.append(text)
        except Exception:
            pass

        wm = self.working.get_context(3)
        if wm:
            parts.append(f"## 最近对话\n{wm}")

        try:
            progress = self.storage.get_all_progress(limit=5)
            if progress:
                lines = ["## 用户研究进展"]
                for p in progress:
                    lines.append(f"- [{p.get('entry_type', '?')}] {p.get('title', '')}: "
                                 f"{p.get('insights', '') or p.get('content', '')[:80]}")
                parts.append("\n".join(lines))
        except Exception:
            pass

        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            unresolved = em.get_unresolved_questions(limit=3)
            if unresolved:
                parts.append("## 上次未解决问题\n" + "\n".join(f"- {q}" for q in unresolved))
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "（首次对话，无语境上下文）"

    def end_session(self):
        summary = self.working.get_session_summary()
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            em.save_session_summary(summary)
        except Exception:
            logger.debug("会话摘要保存失败", exc_info=True)
        self.working.clear()

    def get_stats(self) -> dict:
        return {"working_memory": self.working.stats()}


# ============================================================
# 辅助函数
# ============================================================

def _exec_tool(tc: dict, tools: list) -> str:
    name = tc.get("name", "")
    for t in tools:
        if t.name == name:
            try:
                result = t.invoke(tc.get("args", {}))
                return str(result) if result else "（空结果）"
            except Exception as e:
                logger.warning("工具 %s 失败: %s", name, e)
                return f"（工具失败: {e}）"
    return f"（未知工具: {name}）"


def _extract_cited_from_messages(messages: list) -> list[dict]:
    import re
    papers = []
    for msg in messages:
        if hasattr(msg, 'content'):
            text = str(msg.content)
            for m in re.finditer(r'\[论文(\d+)\]\s+(.+)', text):
                papers.append({"index": int(m.group(1)), "title": m.group(2)[:100]})
    return papers
