"""智能问答服务 — Agent + 工具调用。

架构变化:
  旧: 输入 → LLM 分类(chat/recall/research/direction) → 固定分支 → 回答
  新: 输入 → LLM + 4 工具(统一从 tools/ 加载) → 自主决策 → 回答

工具来源: tools/kb.py, tools/search.py, tools/history.py
  Agent 和 LangGraph 共用同一套工具，不再各自定义闭包。
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..utils import get_llm
from ..core.storage import PerUserStorage
from ..rag.retrieval import HybridRetriever
from ..memory.working import WorkingMemory
from ..tools import make_all_agent_tools

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是科研助手，帮助用户进行学术研究。你有以下工具可以调用:

## 工具说明

1. **query_knowledge_base(query, limit)** — 搜索本地论文知识库
   适用: 科研问题、论文数据查询、领域进展

2. **query_progress(query, limit)** — 查询用户研究进展记录
   适用: "我做过什么实验"、"进展如何"

3. **recall_history(query, limit)** — 搜索历史会话讨论
   适用: "上次讨论了什么"、"之前分析过XX"

4. **search_papers_online(query)** — 外部搜索新论文(ArXiv + S2)
   适用: "找一下XX的最新论文"、"搜索XX领域"

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


class QAService:
    """智能问答服务 — Agent 驱动，工具统一从 tools/ 加载。"""

    def __init__(self, username: str, storage: PerUserStorage):
        self.username = username
        self.storage = storage
        self.retriever = HybridRetriever(storage, username)
        self.working = WorkingMemory(username, str(storage.user_dir))

    # ── 主入口 ──

    def ask(self, question: str) -> dict:
        """Agent 循环: LLM + 工具自主调用。

        Returns:
            {"answer": str, "cited_papers": [...], "tool_calls": [...]}
        """
        tools = make_all_agent_tools(self.storage, self.username)
        context = self._build_context()

        llm = get_llm(temperature=0.3, max_tokens=2048)
        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]

        tool_calls_log = []
        cited_papers = []

        for _ in range(3):
            response = llm_with_tools.invoke(messages)

            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                messages.append(response)
                break

            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                logger.info("Agent 调用工具: %s(%s)", tool_name,
                            {k: str(v)[:80] for k, v in tool_args.items()})

                tool_result = _exec_tool(tc, tools)
                tool_calls_log.append({
                    "tool": tool_name, "args": tool_args,
                    "result_len": len(str(tool_result)),
                })

                if tool_name == "query_knowledge_base":
                    cited_papers.extend(_extract_cited(str(tool_result)))

                messages.append(ToolMessage(
                    content=str(tool_result), tool_call_id=tool_id,
                ))

        final_msg = messages[-1]
        answer = str(final_msg.content) if hasattr(final_msg, 'content') else ""
        self.working.add_turn(question, answer, cited_papers, "")

        return {
            "answer": answer,
            "cited_papers": cited_papers,
            "tool_calls": tool_calls_log,
        }

    # ── 上下文构建 ──

    def _build_context(self) -> str:
        """构建 Agent 上下文（用户画像 + 工作记忆 + 进展摘要）。"""
        parts = []

        # 用户画像
        try:
            from ..memory.user_profile import UserProfileManager
            profile_mgr = UserProfileManager(self.username, str(self.storage.user_dir))
            profile_text = profile_mgr.build_context_for_qa()
            if profile_text:
                parts.append(profile_text)
        except Exception:
            logger.debug("用户画像注入跳过", exc_info=True)

        # 工作记忆
        wm = self.working.get_context(3)
        if wm:
            parts.append(f"## 最近对话\n{wm}")

        # 进展摘要
        try:
            progress = self.storage.get_all_progress(limit=5)
            if progress:
                lines = ["## 用户研究进展"]
                for p in progress:
                    lines.append(
                        f"- [{p.get('entry_type', '?')}] {p.get('title', '')}: "
                        f"{p.get('insights', '') or p.get('content', '')[:80]}"
                    )
                parts.append("\n".join(lines))
        except Exception:
            pass

        # 未解决问题
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            unresolved = em.get_unresolved_questions(limit=3)
            if unresolved:
                lines = ["## 上次未解决问题"]
                for q in unresolved:
                    lines.append(f"- {q}")
                parts.append("\n".join(lines))
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "（首次对话，无语境上下文）"

    # ── 会话管理 ──

    def end_session(self):
        """结束会话: 刷盘事实 + 生成摘要。"""
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


# ── 工具执行辅助（模块级，qa/graph 共用）──

def _exec_tool(tc: dict, tools: list) -> str:
    """执行工具调用，返回结果文本。"""
    name = tc.get("name", "")
    for t in tools:
        if t.name == name:
            try:
                result = t.invoke(tc.get("args", {}))
                return str(result) if result else "（空结果）"
            except Exception as e:
                logger.warning("工具 %s 失败: %s", name, e)
                return f"（工具执行失败: {e}）"
    return f"（未知工具: {name}）"


def _extract_cited(text: str) -> list[dict]:
    """从工具返回文本提取 [论文N] 引用。"""
    import re
    papers = []
    for m in re.finditer(r'\[论文(\d+)\]\s+(.+)', text):
        papers.append({"index": int(m.group(1)), "title": m.group(2)[:100]})
    return papers
