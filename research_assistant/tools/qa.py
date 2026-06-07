"""智能问答服务 — 意图识别 → 按需检索 → 自适应回答。

流程:
  Phase 1: LLM 意图识别 (每次1次调用, ~0.3s)
    输入: 工作记忆 + 进展摘要 + 未解决问题 + 用户输入
    输出: intent(chat|research|direction) + keywords + understanding

  Phase 2:
    chat      → Phase1 已给出回答，直接返回
    research  → 检索论文/进展/图谱 → LLM 生成回答
    direction → 同上 + 提取后续科研方向 + 可记录到progress

三种意图自适应输出:
  chat:     简短, 无引用, 无方向
  research: 有据可查, 标注引用, 无方向
  direction: 有据可查, 标注引用, 2-3个后续方向
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils import get_llm, extract_json_from_llm_response
from ..core.storage import PerUserStorage
from ..rag.retrieval import HybridRetriever
from ..memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

INTENT_PROMPT = """你是科研助手的意图识别模块。

## 四种意图
- chat: 闲聊、问候、概念解释，不需要检索任何数据
- recall: 用户询问历史/过去的讨论内容，需要检索情景记忆
- research: 需要查论文/进展/知识图谱才能回答的科研问题
- direction: 科研问题 + 用户想知道后续方向/下一步怎么做

## 判断规则（优先级从高到低）
1. 问候/寒暄/自我介绍/感谢/再见 → chat
2. 简单概念解释（什么是XX、XX的定义）→ chat
3. 询问过去讨论的内容（"上次讨论了什么""之前聊过XX""还记得XX吗""回顾一下XX"）→ recall
4. 需要查论文才能回答的科研问题 + 明确说"下一步/后续/接下来/建议方向" → direction
5. 需要查论文才能回答的科研问题 → research

## keywords 提取规则
- 从用户问题中提取**具体术语**作为检索词
- recall: 提取用户想回顾的主题词（"上次讨论的CVD温度" → ["CVD", "温度"]）
- research/direction: 提取学术术语，含中英文同义词
- 中文学术论文常用词: 结论、工艺、参数、最佳、最优、影响、分析

## 背景上下文
{context}

## 用户输入
{question}

## 输出 JSON
{{
  "intent": "chat|recall|research|direction",
  "keywords": ["检索词"],
  "understanding": "意图理解（1句话）",
  "answer": "chat时直接回答（1-3句），其他时留空"
}}

只返回 JSON。"""

RESEARCH_ANSWER_PROMPT = """你是科研助手，基于检索结果回答用户问题。

## 用户意图
{understanding}

## 对话上下文
{working_context}

## 检索到的论文
{papers_text}

## 相关实验进展
{progress_text}

## 知识图谱
{kg_text}

## 历史相关讨论
{episodic_text}

## 用户问题
{question}

## 回答要求
- 从"匹配内容"中提取**具体定量信息**（数值、参数、指标、性能数据等），不要只说"未提供"
- 回答基于检索结果，不凭空编造，不推断缺失的数据
- 引用论文时用 [论文N] 标注
- 如果有矛盾信息，明确指出
- {direction_hint}

用 Markdown 格式输出。"""

DIRECTION_EXTRACT_PROMPT = """从以下回答中提取 2-3 个后续科研方向。

回答:
{answer}

输出 JSON:
{{
  "directions": [
    {{
      "title": "方向简述(<30字)",
      "description": "具体描述(1句话)",
      "priority": "high|medium|low",
      "suggested_action": "建议的具体行动"
    }}
  ]
}}

只返回 JSON。"""


class QAService:
    """智能问答服务。"""

    def __init__(self, username: str, storage: PerUserStorage):
        self.username = username
        self.storage = storage
        self.retriever = HybridRetriever(storage, username)
        self.working = WorkingMemory(username, str(storage.user_dir))

    # ── 主入口 ──

    def ask(self, question: str, record_directions: bool = True) -> dict:
        """执行智能问答。

        Returns:
            {"intent": str, "answer": str, "cited_papers": [...],
             "directions": [...], "progress_recorded": bool}
        """
        # === Phase 1: 意图识别 ===
        intent_result = self._recognize_intent(question)

        intent = intent_result["intent"]
        understanding = intent_result.get("understanding", "")

        # === 按意图路由检索 ===
        keywords = intent_result.get("keywords", [question])

        if intent == "chat":
            # 闲聊：不检索，直接回答
            answer = intent_result.get("answer", "")
            if not answer:
                answer = self._chat_answer(question, understanding)
            self.working.add_turn(question, answer, [], understanding)
            return {
                "intent": "chat",
                "answer": answer,
                "cited_papers": [],
                "directions": [],
                "progress_recorded": False,
            }

        if intent == "recall":
            # 回忆：只检索情景记忆，不检索论文/进展/图谱
            episodic_context = self._retrieve_episodic(keywords)
            answer = self._recall_answer(question, understanding, episodic_context)
            self.working.add_turn(question, answer, [], understanding)
            return {
                "intent": "recall",
                "answer": answer,
                "cited_papers": [],
                "directions": [],
                "progress_recorded": False,
            }

        # === research / direction: 全检索 ===
        papers = self._retrieve_papers(keywords)
        progress = self._retrieve_progress(keywords)
        kg = self._retrieve_knowledge_graph(keywords)

        # 情景记忆
        if intent == "direction":
            episodic_context = self._retrieve_episodic(keywords)
        else:
            episodic_context = self._retrieve_episodic(keywords) if len(papers) < 3 else ""

        # 构建回答
        direction_flag = (intent == "direction")
        answer = self._generate_answer(
            question=question,
            understanding=understanding,
            papers=papers,
            progress=progress,
            kg=kg,
            episodic_context=episodic_context,
            with_direction=direction_flag,
        )

        # 提取引用
        cited = self._extract_cited_papers(answer, papers)

        # 更新工作记忆
        self.working.add_turn(question, answer, cited, understanding)

        # 提取方向
        directions = []
        progress_recorded = False
        if direction_flag:
            directions = self._extract_directions(answer)
            if record_directions and directions:
                progress_recorded = self._record_directions(question, directions)

        return {
            "intent": intent,
            "answer": answer,
            "cited_papers": cited,
            "directions": directions,
            "progress_recorded": progress_recorded,
        }

    # ── Phase 1: 意图识别 ──

    def _recognize_intent(self, question: str) -> dict:
        """LLM 语义识别意图。所有输入都走 LLM，不硬编码规则。"""

        context = self._build_intent_context()
        llm = get_llm(temperature=0.1, max_tokens=256)

        try:
            response = llm.invoke([
                SystemMessage(content="你是意图识别模块。只返回 JSON。"),
                HumanMessage(content=INTENT_PROMPT.format(
                    context=context, question=question,
                )),
            ])
            result = extract_json_from_llm_response(str(response.content))
            return {
                "intent": result.get("intent", "chat"),
                "keywords": result.get("keywords", []),
                "understanding": result.get("understanding", ""),
                "answer": result.get("answer", ""),
            }
        except Exception:
            logger.debug("意图识别失败，降级为 chat", exc_info=True)
            return {
                "intent": "chat",
                "keywords": [],
                "understanding": "",
                "answer": "",
            }

    def _build_intent_context(self) -> str:
        """构建 Phase 1 的上下文（不含检索结果）。

        改进: 注入用户画像（被动注入机制，参考 DeerFlow DynamicContextMiddleware）。
        """
        parts = []

        # ── 用户画像（新增：被动注入）──
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
                parts.append("## 用户研究进展")
                for p in progress:
                    parts.append(
                        f"- [{p.get('entry_type','?')}] {p.get('title','')}: "
                        f"{p.get('insights','') or p.get('content','')[:80]}"
                    )
        except Exception:
            pass

        # 未解决问题
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            unresolved = em.get_unresolved_questions(limit=3)
            if unresolved:
                parts.append("## 上次未解决问题")
                for q in unresolved:
                    parts.append(f"- {q}")
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "（首次对话，无语境上下文）"

    # ── 检索子模块 ──

    def _retrieve_papers(self, keywords: list[str],
                          limit: int = 10, use_expansion: bool = True) -> list[dict]:
        query = " ".join(keywords) if keywords else ""
        if not query:
            return []
        try:
            if use_expansion:
                # MQE + HyDE 扩展检索
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
        query = " ".join(keywords) if keywords else ""
        if not query:
            return []
        try:
            return self.retriever.search_progress(query, limit=limit)
        except Exception:
            logger.debug("进展检索失败", exc_info=True)
            return []

    def _retrieve_knowledge_graph(self, keywords: list[str]) -> list[str]:
        try:
            from ..memory.semantic import SemanticMemory
            sm = SemanticMemory(self.username)
            if not sm.available:
                sm.close()
                return []
            results = []
            for kw in keywords[:3]:
                if len(kw) < 2:
                    continue
                neighbors = sm.search_entity(kw)
                for n in neighbors[:3]:
                    results.append(
                        f"[{n.get('type','')}] {n.get('entity','')} "
                        f"--({n.get('relation','')})--"
                    )
            sm.close()
            return results[:10]
        except Exception:
            logger.debug("图谱检索失败", exc_info=True)
            return []

    def _recall_answer(self, question: str, understanding: str,
                        episodic_context: str) -> str:
        """基于情景记忆检索结果回答历史相关问题。

        仅检索情景记忆，不检索论文/进展/图谱，避免无意义检索。
        """
        if not episodic_context:
            return (
                "抱歉，我暂时没有找到相关的历史讨论记录。"
                "你可以尝试用 recall 命令搜索，或告诉我具体的主题帮"
                "我定位。"
            )
        try:
            llm = get_llm(temperature=0.3, max_tokens=512)
            prompt = (
                f"用户问: {question}\n\n"
                f"意图: {understanding}\n\n"
                f"以下是之前会话中相关的讨论记录（按语义相关度和时间排序）:\n"
                f"{episodic_context}\n\n"
                f"请基于这些历史记录回答用户的问题。"
                f"如果记录不足以完整回答，请诚实告知。"
            )
            response = llm.invoke([
                SystemMessage(content="你是科研助手。基于历史记录回答用户关于过往讨论的问题。"),
                HumanMessage(content=prompt),
            ])
            return str(response.content).strip()
        except Exception:
            return "抱歉，检索历史讨论时出现错误。请稍后重试。"

    def _retrieve_episodic(self, keywords: list[str]) -> str:
        """检索相关历史会话摘要（语义相似度 × 时间衰减）。

        仅在 direction 意图和 papers 不足时调用。
        使用 recall_context_hybrid 混合检索，
        新近讨论权重更高。
        """
        query = " ".join(keywords) if keywords else ""
        if not query:
            return ""
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            results = em.recall_context_hybrid(query, limit=3, decay_days=30.0)
            if results:
                parts = []
                for r in results:
                    text = (r.get("text", "") or r.get("payload", {}).get("text", ""))[:150]
                    date_str = r.get("payload", {}).get("session_date", "")
                    if text:
                        parts.append(f"- [{date_str}] {text}")
                return "\n".join(parts)
        except Exception:
            logger.debug("情景记忆检索失败", exc_info=True)
        return ""

    def _chat_answer(self, question: str, context_hint: str = "") -> str:
        """生成简短闲聊回答（不检索）。"""
        try:
            llm = get_llm(temperature=0.5, max_tokens=128)
            prompt = f"用户说: {question}\n请简短友好地回复(1-2句话)，介绍自己是科研助手。"
            response = llm.invoke([
                SystemMessage(content="你是友好的科研助手。回复简短自然。"),
                HumanMessage(content=prompt),
            ])
            return str(response.content).strip()
        except Exception:
            return "你好！我是科研助手，可以帮你检索知识库论文、解答科研问题。有什么需要？"

    # ── Phase 2: 生成回答 ──

    def _generate_answer(self, question: str, understanding: str,
                          papers: list[dict], progress: list[dict],
                          kg: list[str], with_direction: bool,
                          episodic_context: str = "") -> str:
        """LLM 基于检索结果生成回答。

        Args:
            episodic_context: 情景记忆混合检索结果（语义×时间），仅
                direction 意图和 papers 不足时传入。
        """
        direction_hint = (
            "最后给出 2-3 个具体的后续科研方向"
            if with_direction else
            "不需要给出后续方向"
        )

        # 构建论文文本：按论文分组，同篇论文的不同章节归入同一编号
        if papers:
            from collections import OrderedDict
            paper_groups = OrderedDict()
            for p in papers:
                pid = str(p.get("paper_id") or p.get("id", ""))
                if not pid:
                    continue
                text = (p.get("text", "") or "")
                payload = p.get("payload", {})
                if not text and isinstance(payload, dict):
                    text = payload.get("text", "") or payload.get("window_text", "")
                if not text:
                    continue  # 跳过空文本结果（BM25 降级等情况）
                if pid not in paper_groups:
                    paper_groups[pid] = {
                        "title": p.get("title", ""),
                        "abstract": (p.get("abstract", "") or "")[:150],
                        "core": p.get("core_claim", ""),
                        "chunks": [],
                    }
                # 后续结果可能补全 title
                if not paper_groups[pid]["title"] and p.get("title"):
                    paper_groups[pid]["title"] = p.get("title")
                if not paper_groups[pid]["core"] and p.get("core_claim"):
                    paper_groups[pid]["core"] = p.get("core_claim")
                paper_groups[pid]["chunks"].append({
                    "heading": p.get("heading_path", ""),
                    "text": text[:800],
                })

            paper_items = list(paper_groups.items())[:5]
            parts = []
            for i, (pid, pinfo) in enumerate(paper_items):
                title = pinfo["title"] or "?"
                parts.append(f"[论文{i+1}] {title}")
                if pinfo["abstract"]:
                    parts.append(f"  摘要: {pinfo['abstract']}")
                if pinfo["core"]:
                    parts.append(f"  核心结论: {pinfo['core']}")
                for chunk in pinfo["chunks"][:3]:
                    hp = f" ({chunk['heading']})" if chunk["heading"] else ""
                    parts.append(f"  匹配内容{hp}: {chunk['text']}")
            papers_text = "\n\n".join(parts)
        else:
            papers_text = "（未检索到相关论文）"

        # 进展
        if progress:
            progress_text = "\n".join(
                f"- [{p.get('entry_type','?')}] {p.get('title','')}: "
                f"{(p.get('content','') or p.get('insights','') or '')[:150]}"
                for p in progress[:5]
            )
        else:
            progress_text = "（无相关进展记录）"

        # 知识图谱
        kg_text = "\n".join(f"- {r}" for r in kg) if kg else "（无相关图谱数据）"

        # 工作记忆
        wm_ctx = self.working.get_context(2)

        llm = get_llm(temperature=0.3, max_tokens=2048)

        prompt = RESEARCH_ANSWER_PROMPT.format(
            understanding=understanding,
            working_context=wm_ctx or "（无最近对话）",
            papers_text=papers_text,
            progress_text=progress_text,
            kg_text=kg_text,
            episodic_text=episodic_context or "（无相关历史讨论）",
            question=question,
            direction_hint=direction_hint,
        )

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研助手。基于检索结果回答，引用论文标注 [论文N]。"),
                HumanMessage(content=prompt),
            ])
            return str(response.content).strip()
        except Exception:
            logger.error("回答生成失败", exc_info=True)
            return "抱歉，回答生成失败。请稍后重试。"

    # ── 后处理 ──

    def _extract_cited_papers(self, answer: str,
                               papers: list[dict]) -> list[dict]:
        """按 paper_id 去重后，匹配 answer 中的 [论文N] 引用。"""
        # 与 _generate_answer 相同的分组顺序
        from collections import OrderedDict
        groups = OrderedDict()
        for p in papers:
            pid = str(p.get("paper_id") or p.get("id", ""))
            if not pid:
                continue
            if pid not in groups or (not groups[pid].get("title") and p.get("title")):
                groups[pid] = p
        cited = []
        for i, (pid, p) in enumerate(groups.items()):
            if f"[论文{i+1}]" in answer:
                cited.append(p)
        return cited

    def _extract_directions(self, answer: str) -> list[dict]:
        try:
            llm = get_llm(temperature=0.1, max_tokens=512)
            response = llm.invoke([
                SystemMessage(content="你是科研方向提取专家。只返回 JSON。"),
                HumanMessage(content=DIRECTION_EXTRACT_PROMPT.format(
                    answer=answer[:3000]
                )),
            ])
            result = extract_json_from_llm_response(str(response.content))
            return result.get("directions", [])
        except Exception:
            logger.debug("方向提取失败", exc_info=True)
            return []

    def _record_directions(self, question: str,
                            directions: list[dict]) -> bool:
        try:
            from ..tools.progress import record_user_progress
            for d in directions:
                title = d.get("title", "未命名")
                content = (
                    f"[来源: Q&A] 问题: {question[:100]}\n"
                    f"{d.get('description','')}\n"
                    f"建议行动: {d.get('suggested_action','')}"
                )
                record_user_progress.invoke({
                    "topic": title,
                    "entry_type": "idea",
                    "title": title,
                    "content": content,
                    "results": d.get("suggested_action", ""),
                    "insights": d.get("description", ""),
                    "timestamp": datetime.now().isoformat(),
                })
            return True
        except Exception:
            logger.debug("progress 记录失败", exc_info=True)
            return False

    # ── 会话管理 ──

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
