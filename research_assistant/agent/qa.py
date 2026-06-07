"""智能问答服务 — Agent 循环 + Hook + 动态工具映射。

核心设计（参考 Claude Code 架构）:
  agent_loop() → 可见的循环体，Hook 挂横切逻辑
  build_system() → 工具描述从定义生成，不硬编码
  hooks → PreToolUse / PostToolUse / BeforeLLM / AfterLLM
"""
from __future__ import annotations

import logging
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
    print(f"\n🤖 分析: {question[:60]}...")

def _tool_log_hook(name: str, args: dict):
    """pre_tool: 显示工具调用。"""
    arg_preview = {k: str(v)[:60] for k, v in args.items()}
    print(f"  🔧 {name}({arg_preview})")
    return None

def _tool_result_hook(_name: str, _args: dict, result: str):
    """post_tool: 显示工具结果摘要。"""
    preview = str(result)[:100].replace("\n", " ")
    print(f"     → {preview}")

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
- 找新论文: 调 search_papers_online
- 可组合调用多个工具
- 工具无结果时诚实告知，不编造

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
    """用 stream 流式输出最终回答，逐 token 打印。

    容错: 如果 llm 没 stream 方法（如 mock）→ 降级为一次性输出。
    """
    import sys

    try:
        stream_method = llm.stream
    except AttributeError:
        # 降级: 直接输出 invoke 结果
        text = str(invoke_response.content) if hasattr(invoke_response, 'content') else str(invoke_response)
        sys.stdout.write(text)
        sys.stdout.flush()
        print()
        messages.append(invoke_response)
        return

    accumulated = []
    try:
        for chunk in stream_method(messages):
            if chunk.content:
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
                accumulated.append(chunk.content)
    except Exception:
        # stream 失败 → 降级
        text = str(invoke_response.content) if hasattr(invoke_response, 'content') else str(invoke_response)
        sys.stdout.write(text)
        sys.stdout.flush()
        accumulated = [text]

    full = "".join(accumulated)
    if full.strip():
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=full))
    else:
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
        self.working = WorkingMemory(username, str(storage.user_dir))

    def ask(self, question: str) -> dict:
        """主入口: 组装 → agent_loop → 回答。"""
        trigger_hooks("agent_start", question)

        # 工具 + 动态 System Prompt
        tools = make_all_agent_tools(self.storage, self.username)
        context = self._build_context()
        system = build_system_prompt(tools, context)

        llm = get_llm(temperature=0.3, max_tokens=2048)
        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=question),
        ]

        # ── Agent 循环 ──
        messages, tool_log = agent_loop(messages, tools, llm_with_tools)

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
