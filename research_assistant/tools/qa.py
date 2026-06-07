"""智能问答服务 — Agent + 工具调用，替代固定意图路由。

架构变化:
  旧: 输入 → LLM 分类(chat/recall/research/direction) → 固定分支 → 回答
  新: 输入 → LLM + 4 工具 → 自主决策调哪些/怎么组合 → 回答

为什么改:
  1. 意图识别是穷人的 tool choice — LLM 被当分类器用，选完走死分支
  2. recall/research/direction 边界模糊，混合场景无法处理
  3. 固定分支意味着每次都要写新的 if-else，而工具调用让 LLM 自己组合
  4. LLM 调用次数没省 (旧: 2-3次, 新: 1+N次按需)

工具清单:
  - query_knowledge_base: 本地论文库语义检索 (Hybrid: BM25 + Dense + RRF)
  - query_progress: 用户研究进展记录查询
  - recall_history: 情景记忆/历史会话检索 (语义×时间衰减)
  - search_papers_online: 外部论文搜索 (ArXiv/Semantic Scholar)

保留:
  - 工作记忆 (WorkingMemory) 集成 — 多轮对话上下文
  - 用户画像被动注入 — 每次回答前注入 user_facts.json
  - 会话管理 (end_session / get_stats)
"""
from __future__ import annotations

import logging
from collections import OrderedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ..utils import get_llm
from ..core.storage import PerUserStorage
from ..rag.retrieval import HybridRetriever
from ..memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# ============================================================
# Agent 系统提示词
# ============================================================

AGENT_SYSTEM_PROMPT = """你是科研助手，帮助用户进行学术研究。你有以下工具可以调用:

## 工具说明

1. **query_knowledge_base(query, limit)** — 搜索本地论文知识库
   适用场景: 用户问科研问题、需要查论文数据、想知道某个领域的研究进展
   返回: 匹配的论文片段（含标题、核心结论、匹配内容）

2. **query_progress(query, limit)** — 查询用户的研究进展记录
   适用场景: 用户问"我做过什么实验"、"进展如何"、"之前记录了哪些"
   返回: 相关的实验/阅读/想法记录

3. **recall_history(query, limit)** — 搜索历史会话讨论
   适用场景: 用户问"上次讨论了什么"、"之前分析过XX"、"还记得吗"
   返回: 历史会话摘要和关键讨论

4. **search_papers_online(query)** — 从外部搜索新论文
   适用场景: 用户想找最新的论文、"搜索一下XX"、"有没有关于XX的新研究"
   返回: 外部搜索到的论文列表

## 决策原则

- 问候/闲聊/概念解释: 不调工具，直接友好回答
- 科研问题(需要数据支撑): 先调 query_knowledge_base，不够再调其他
- 回忆历史: 调 recall_history
- 找新论文/外部搜索: 调 search_papers_online
- 可以组合调用多个工具，比如同时查论文和进展
- 工具返回无结果时诚实告知，不要编造数据

## 回答格式

- 基于工具返回的真实内容回答，引用论文用 [论文标题] 标注
- 包含具体数值、参数、指标（如果工具返回里有的话）
- 如果工具返回了论文，在末尾列出引用的论文
- 中文回答，学术风格但可读

## 背景上下文

{context}"""


class QAService:
    """智能问答服务 — Agent 驱动，工具自主调用。"""

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
        # 构建工具（闭包捕获 self）
        tools = self._build_tools()

        # 构建上下文（用户画像 + 工作记忆 + 进展摘要）
        context = self._build_context()

        # Agent 循环
        llm = get_llm(temperature=0.3, max_tokens=2048)
        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]

        tool_calls_log = []
        cited_papers = []

        # 最多 3 轮工具调用（防止死循环）
        for _ in range(3):
            response = llm_with_tools.invoke(messages)

            # 无工具调用 → LLM 直接回答 → 结束
            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                messages.append(response)
                break

            # 有工具调用 → 执行 → 结果反馈
            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                logger.info(
                    "Agent 调用工具: %s(%s)", tool_name,
                    {k: str(v)[:80] for k, v in tool_args.items()},
                )

                # 执行工具
                tool_result = self._execute_tool(tool_name, tool_args, tools)
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_len": len(str(tool_result)),
                })

                # 收集引用的论文
                if tool_name == "query_knowledge_base":
                    cited_papers.extend(
                        self._extract_cited_from_result(tool_result)
                    )

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                ))

        # 提取最终回答
        final_msg = messages[-1]
        answer = str(final_msg.content) if hasattr(final_msg, 'content') else ""

        # 更新工作记忆
        self.working.add_turn(question, answer, cited_papers, "")

        return {
            "answer": answer,
            "cited_papers": cited_papers,
            "tool_calls": tool_calls_log,
        }

    # ── 工具构建 ──

    def _build_tools(self) -> list:
        """构建 Agent 可调用的工具列表（闭包捕获 self）。"""
        _self = self

        @tool
        def query_knowledge_base(query: str, limit: int = 5) -> str:
            """搜索本地论文知识库。用来回答科研问题、查找论文中的具体参数/方法/结论。

            Args:
                query: 搜索关键词（建议用学术术语，如 "CVD温度优化" "钙钛矿PLQY"）
                limit: 返回结果数，默认5
            """
            papers = _self._retrieve_papers(
                [query], limit=limit, use_expansion=True,
            )
            if not papers:
                return "（未在本地知识库中找到相关论文）"
            return _self._format_papers_for_llm(papers)

        @tool
        def query_progress(query: str, limit: int = 5) -> str:
            """查询用户的研究进展记录。用来回答"我做过什么实验""有什么进展""记录了哪些想法"。

            Args:
                query: 搜索关键词
                limit: 返回结果数，默认5
            """
            results = _self._retrieve_progress([query], limit=limit)
            if not results:
                return "（未找到相关进展记录）"
            return _self._format_progress_for_llm(results)

        @tool
        def recall_history(query: str, limit: int = 3) -> str:
            """搜索历史会话和讨论记录。用来回答"上次讨论了XX""之前分析过YY""还记得吗"。

            Args:
                query: 搜索关键词（提取你想回顾的主题词）
                limit: 返回结果数，默认3
            """
            results = _self._retrieve_episodic([query])
            if not results:
                return "（未找到相关历史讨论记录）"
            # 截取前 limit 条
            parts = results.strip().split("\n- [")
            if len(parts) > limit:
                results = "\n- [".join(parts[:limit + 1])
            return results

        @tool
        def search_papers_online(query: str) -> str:
            """从外部学术平台搜索新论文（ArXiv + Semantic Scholar）。
            用来回答"找一下XX的最新论文""搜索XX领域""有没有关于XX的研究"。

            Args:
                query: 搜索关键词（英文效果更好，如 "diamond CVD temperature optimization"）
            """
            try:
                from ..tools.search_orchestrator import SearchOrchestrator
                orch = SearchOrchestrator(_self.storage, _self.username)
                result = orch.search(query, sources=["arxiv", "s2"])
                papers = result.get("papers", [])
                if not papers:
                    return (
                        f"（外部搜索未找到相关论文。搜索策略: "
                        f"{result.get('search_focus', '')}）"
                    )
                lines = [
                    f"外部搜索: 找到 {result.get('total_found', 0)} 篇, "
                    f"精选 Top-{len(papers)}, 耗时 {result.get('duration_sec', 0):.1f}s\n",
                ]
                for i, p in enumerate(papers[:5], 1):
                    lines.append(
                        f"[{i}] {p.get('title', '?')}\n"
                        f"    评分: {p.get('composite_score', 0):.0f}/100 | "
                        f"引用: {p.get('citation_count', 0) or 0} | "
                        f"年份: {p.get('year', '')}\n"
                        f"    核心: {p.get('core_contribution', '')[:200]}\n"
                        f"    摘要: {(p.get('abstract', '') or '')[:300]}"
                    )
                return "\n".join(lines)
            except Exception:
                logger.debug("外部搜索失败", exc_info=True)
                return "（外部搜索暂时不可用，请稍后重试）"

        return [
            query_knowledge_base,
            query_progress,
            recall_history,
            search_papers_online,
        ]

    def _execute_tool(self, name: str, args: dict, tools: list) -> str:
        """执行工具调用并返回结果文本。"""
        for t in tools:
            if t.name == name:
                try:
                    result = t.invoke(args)
                    return str(result) if result else "（工具返回空结果）"
                except Exception as e:
                    logger.warning("工具 %s 执行失败: %s", name, e)
                    return f"（工具执行失败: {e}）"
        return f"（未知工具: {name}）"

    # ── 上下文构建 ──

    def _build_context(self) -> str:
        """构建 Agent 上下文（注入用户画像 + 工作记忆 + 进展摘要）。

        替代旧的 _build_intent_context() — 内容相同但不再为"意图识别"服务，
        而是作为 Agent 的背景知识，帮助它更好地理解用户问题。
        """
        parts = []

        # 用户画像（被动注入）
        try:
            from ..memory.user_profile import UserProfileManager
            profile_mgr = UserProfileManager(
                self.username, str(self.storage.user_dir),
            )
            profile_text = profile_mgr.build_context_for_qa()
            if profile_text:
                parts.append(profile_text)
        except Exception:
            logger.debug("用户画像注入跳过", exc_info=True)

        # 工作记忆（最近 3 轮对话）
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

    # ── 检索子模块（工具实现）──

    def _retrieve_papers(self, keywords: list[str],
                         limit: int = 10, use_expansion: bool = True) -> list[dict]:
        """MQE + HyDE 扩展检索本地论文库。"""
        query = " ".join(keywords) if keywords else ""
        if not query:
            return []
        try:
            if use_expansion:
                return self.retriever.search_papers_expanded(
                    query, limit=limit,
                    enable_mqe=True, mqe_expansions=3,
                    enable_hyde=True,
                )
            return self.retriever.search_papers(query, limit=limit)
        except Exception:
            logger.debug("论文检索失败", exc_info=True)
            return []

    def _retrieve_progress(self, keywords: list[str],
                           limit: int = 5) -> list[dict]:
        """检索用户研究进展记录。"""
        query = " ".join(keywords) if keywords else ""
        if not query:
            return []
        try:
            return self.retriever.search_progress(query, limit=limit)
        except Exception:
            logger.debug("进展检索失败", exc_info=True)
            return []

    def _retrieve_episodic(self, keywords: list[str]) -> str:
        """检索情景记忆（语义相似度 × 时间衰减）。"""
        query = " ".join(keywords) if keywords else ""
        if not query:
            return ""
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            results = em.recall_context_hybrid(
                query, limit=3, decay_days=30.0,
            )
            if results:
                parts = []
                for r in results:
                    text = (
                        r.get("text", "") or
                        r.get("payload", {}).get("text", "")
                    )[:150]
                    date_str = r.get("payload", {}).get("session_date", "")
                    if text:
                        parts.append(f"- [{date_str}] {text}")
                return "\n".join(parts)
        except Exception:
            logger.debug("情景记忆检索失败", exc_info=True)
        return ""

    # ── 工具结果格式化 ──

    def _format_papers_for_llm(self, papers: list[dict]) -> str:
        """将检索到的论文格式化为 LLM 友好的 Markdown 文本。

        按 paper_id 分组，同篇论文的不同 chunk 归入同一编号。
        """
        if not papers:
            return "（未检索到相关论文）"

        # 按 paper_id 分组
        groups = OrderedDict()
        for p in papers:
            pid = str(p.get("paper_id") or p.get("id", ""))
            if not pid:
                continue
            text = (p.get("text", "") or
                    p.get("payload", {}).get("text", "") or
                    p.get("payload", {}).get("window_text", ""))
            if not text:
                continue
            if pid not in groups:
                groups[pid] = {
                    "title": p.get("title", ""),
                    "abstract": (p.get("abstract", "") or "")[:150],
                    "core": p.get("core_claim", ""),
                    "chunks": [],
                }
            if not groups[pid]["title"] and p.get("title"):
                groups[pid]["title"] = p.get("title")
            if not groups[pid]["core"] and p.get("core_claim"):
                groups[pid]["core"] = p.get("core_claim")
            groups[pid]["chunks"].append({
                "heading": p.get("heading_path", ""),
                "text": text[:800],
            })

        # 限制总 Token（最多 5 篇论文，每篇 3 个 chunk）
        items = list(groups.items())[:5]
        lines = []
        for i, (pid, info) in enumerate(items):
            title = info["title"] or "?"
            lines.append(f"\n### [论文{i + 1}] {title}")
            if info["abstract"]:
                lines.append(f"摘要: {info['abstract']}")
            if info["core"]:
                lines.append(f"核心结论: {info['core']}")
            for chunk in info["chunks"][:3]:
                hp = f" ({chunk['heading']})" if chunk["heading"] else ""
                lines.append(f"匹配内容{hp}: {chunk['text']}")

        return "\n".join(lines)

    def _format_progress_for_llm(self, entries: list[dict]) -> str:
        """格式化进展记录为文本。"""
        lines = []
        for i, p in enumerate(entries[:5], 1):
            payload = p.get("payload", {})
            title = payload.get("title", p.get("title", "?"))
            content = (
                payload.get("content", "") or
                payload.get("text", "") or
                p.get("text", "")
            )[:200]
            entry_type = payload.get("entry_type", p.get("type", "?"))
            lines.append(f"[{i}] [{entry_type}] {title}: {content}")
        return "\n".join(lines) if lines else "（无相关进展记录）"

    def _extract_cited_from_result(self, tool_result: str) -> list[dict]:
        """从工具返回文本中提取论文引用信息。"""
        papers = []
        # 简单解析: 匹配 [论文N] 标记
        import re
        for m in re.finditer(r'\[论文(\d+)\]\s+(.+)', tool_result):
            papers.append({
                "index": int(m.group(1)),
                "title": m.group(2)[:100],
            })
        return papers

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
        """获取会话统计。"""
        return {"working_memory": self.working.stats()}
