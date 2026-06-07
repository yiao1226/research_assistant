"""科研助手 LangGraph 工作流 — Agent 驱动，3 路径，4 节点。

架构变化:
  旧: 9 个固定节点串行，写死 Top-3深/4-8轻/9+跳，不可迭代
  新: 4 个 Agent 节点，LLM 自主决策搜索和分析策略

三条工作流:
  /review          → understand → research(迭代) → synthesize_review → user_review
  /research        → understand → research(迭代) → synthesize_answer  → END
  /progress-report → understand → research(迭代) → synthesize_report  → user_review

核心创新:
  - research 节点是可迭代的搜索-评估循环（Agent 不满意就换策略重搜）
  - 每个节点是 Agent + 工具，不是单次 LLM 调用
  - 节点间共享状态，user_review 支持人机协同修改
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Literal

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph.config import get_config

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from .utils import get_llm
from .state import ResearchState, get_runtime_context

# ============================================================
# Agent 系统提示词
# ============================================================

UNDERSTAND_SYSTEM = """你是科研策略规划专家。你的任务是在搜索之前先盘点已有的知识基础。

你有以下工具:
- query_knowledge_base: 搜索本地论文知识库，了解已有文献
- query_progress: 查看用户的研究进展记录
- recall_history: 搜索历史会话讨论

你需要产出:
1. 已有的知识基础: KB有几篇相关论文、进展记录了哪些实验、历史讨论过什么
2. 搜索计划: 应该搜什么关键词（中英文各2-3个）、重点方向、搜索策略

根据盘点结果制定搜索策略，不要盲目搜索。"""

RESEARCH_SYSTEM = """你是学术文献搜索分析专家。搜索新论文，评估质量，决定是否继续搜索。

你有以下工具:
- search_papers_online: 从 ArXiv + Semantic Scholar 搜索新论文
- query_knowledge_base: 搜索本地论文库（补充外部搜索）

搜索策略:
1. 先用核心关键词搜索，看结果质量和覆盖度
2. 如果结果偏少或方向偏差，换角度再搜
3. 如果综述论文太多缺少方法细节，增加 method/fabrication 等关键词
4. 满意后输出评估: 共找到X篇，覆盖了哪些方向，质量如何

评估标准:
- 论文数量 >= 5 且方向覆盖完整 → 满意
- 缺少方法细节 → 需要补充搜索
- 结果全是综述 → 需要搜具体研究方向

最多建议 3 轮搜索。输出你的评估决定。"""

SYNTHESIZE_REVIEW_SYSTEM = """你是学术综述撰写专家。基于搜索到的论文，撰写结构完整的文献综述。

综述结构:
## 摘要 (200字以内)
## 1. 引言 (研究背景和意义)
## 2. 研究脉络与分类 (主要方向和代表工作)
## 3. 核心方法与技术对比 (表格形式)
## 4. 关键发现与趋势
## 5. 结论与展望

要求:
- 所有引用来自搜索到的真实论文，用 [N] 标注
- 中文撰写，学术风格
- 2500-4000字
- 在综述末尾列出参考文献"""

SYNTHESIZE_ANSWER_SYSTEM = """你是科研问答专家。基于搜索和分析结果，深度回答用户问题。

要求:
- 基于搜索到的论文和进展数据，给出有据可查的回答
- 引用论文用 [论文标题] 标注
- 如果用户想知道后续方向，给出2-3个具体建议
- 中文回答，包含具体数值和参数"""

SYNTHESIZE_REPORT_SYSTEM = """你是科研进展评估专家。综合文献库和用户进展记录，生成评估报告。

报告结构:
## 一、文献覆盖度 (已有论文覆盖了哪些方向，缺什么)
## 二、用户实验进展 (完成了哪些实验，关键结果)
## 三、知识缺口分析 (重要但未覆盖的方向)
## 四、文献对照 (用户结果 vs 文献最优值)
## 五、下一步建议 (具体可执行的下一步)

要求: 中文，基于数据说话，建议具体可执行。"""

# ============================================================
# 节点 ① — understand: 盘点现状 + 制定策略
# ============================================================

def node_understand(state: ResearchState) -> dict[str, Any]:
    """Agent 节点: 盘点 KB + 进度 + 历史，制定搜索策略。

    调工具自主了解已有基础，不盲目搜索。
    """
    topic = state.get("topic", "")
    wf_type = state.get("workflow_type", "review")
    thread_id = _get_thread_id()
    ctx = get_runtime_context(thread_id)
    storage = ctx.get("_storage")
    username = ctx.get("_username", "default")

    llm = get_llm(temperature=0.2, max_tokens=600)
    tools = _build_understand_tools(storage, username)
    llm_with_tools = llm.bind_tools(tools)

    wf_labels = {"review": "文献综述", "research": "深度研究", "progress_report": "进展评估"}
    wf_label = wf_labels.get(wf_type, "深度分析")

    messages = [
        SystemMessage(content=UNDERSTAND_SYSTEM),
        HumanMessage(content=f"研究主题: {topic}\n工作流类型: {wf_label}\n\n请先盘点和主题相关的已有基础，然后制定搜索计划。"),
    ]

    plan_text = ""
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            plan_text = str(response.content)
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = _exec_tool(tc, tools)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc.get("id", "")))
        response = llm_with_tools.invoke(messages)
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            plan_text = str(response.content)
            break

    return {
        "search_plan": plan_text,
        "current_stage": "understand",
    }


def _build_understand_tools(storage, username: str):
    """understand 节点的工具: 只查已有数据，不需要外部搜索。"""
    from langchain_core.tools import tool

    @tool
    def query_knowledge_base(query: str, limit: int = 5) -> str:
        """搜索本地论文库，了解已有文献。"""
        try:
            from .rag.retrieval import HybridRetriever
            retriever = HybridRetriever(storage, username)
            papers = retriever.search_papers_expanded(query, limit=limit, enable_mqe=False, enable_hyde=False)
            if not papers:
                return "（本地KB无相关论文）"
            items = []
            seen = set()
            for p in papers[:limit]:
                title = p.get("title", "?")
                if title in seen:
                    continue
                seen.add(title)
                items.append(f"- {title}\n  核心: {p.get('core_claim', '')[:100]}")
            return "\n".join(items) if items else "（无结果）"
        except Exception as e:
            return f"（KB搜索失败: {e}）"

    @tool
    def query_progress(query: str, limit: int = 5) -> str:
        """查看用户的研究进展记录。"""
        try:
            retriever = HybridRetriever(storage, username)
            results = retriever.search_progress(query, limit=limit)
            if not results:
                return "（无相关进展记录）"
            items = []
            for p in results[:limit]:
                payload = p.get("payload", {})
                items.append(f"- [{payload.get('entry_type', '?')}] {payload.get('title', '?')}: {payload.get('content', payload.get('text', ''))[:120]}")
            return "\n".join(items)
        except Exception as e:
            return f"（进展查询失败: {e}）"

    @tool
    def recall_history(query: str, limit: int = 3) -> str:
        """搜索历史会话讨论记录。"""
        try:
            from .memory.episodic import EpisodicMemory
            em = EpisodicMemory(storage, username)
            results = em.recall_context_hybrid(query, limit=limit)
            if not results:
                return "（无相关历史讨论）"
            items = []
            for r in results[:limit]:
                text = r.get("text", "") or r.get("payload", {}).get("text", "")[:120]
                if text:
                    items.append(f"- {text}")
            return "\n".join(items) if items else "（无结果）"
        except Exception:
            return "（历史检索失败）"

    return [query_knowledge_base, query_progress, recall_history]


# ============================================================
# 节点 ② — research: 搜索分析循环 ★ 核心
# ============================================================

def node_research(state: ResearchState) -> dict[str, Any]:
    """Agent 节点: 迭代搜索 + 分析 + 自评估。

    每次 LLM 调工具搜索 → 看结果 → 判断是否满意。
    不满意就换策略重搜，最多3轮。
    """
    topic = state.get("topic", "")
    plan = state.get("search_plan", "")
    iteration = state.get("search_iterations", 0) + 1
    papers = list(state.get("papers_found", []))
    thread_id = _get_thread_id()
    ctx = get_runtime_context(thread_id)
    storage = ctx.get("_storage")
    username = ctx.get("_username", "default")

    llm = get_llm(temperature=0.3, max_tokens=1024)
    tools = _build_research_tools(storage, username)
    llm_with_tools = llm.bind_tools(tools)

    prev_papers_summary = f"已有 {len(papers)} 篇论文" if papers else "尚无搜索结果"
    prev_titles = "\n".join(f"- {p.get('title', '?')[:80]}" for p in papers[-10:]) if papers else ""

    iteration_msg = (
        f"研究主题: {topic}\n\n"
        f"搜索计划: {plan}\n\n"
        f"第 {iteration} 轮搜索\n"
        f"之前已找到: {prev_papers_summary}\n{prev_titles}\n\n"
        f"请搜索论文，然后评估结果质量和覆盖度。"
    )

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=iteration_msg),
    ]

    result_text = ""
    new_papers = []

    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            result_text = str(response.content)
            break
        messages.append(response)
        for tc in response.tool_calls:
            tool_result = _exec_tool(tc, tools)
            if tc["name"] == "search_papers_online":
                parsed = _parse_search_result(tool_result)
                new_papers.extend(parsed)
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc.get("id", "")))

    # 去重合并
    seen = {p.get("title", "") for p in papers}
    for p in new_papers:
        if p.get("title", "") not in seen:
            seen.add(p.get("title", ""))
            papers.append(p)

    # LLM 自我评估是否满意
    satisfied = _check_agent_satisfied(result_text, len(papers), iteration)

    return {
        "papers_found": papers,
        "search_iterations": iteration,
        "agent_satisfied": satisfied,
        "search_summary": result_text[:500],
        "current_stage": "research",
    }


def _build_research_tools(storage, username: str):
    """research 节点的工具: 外部搜索 + 本地KB。"""
    from langchain_core.tools import tool

    @tool
    def search_papers_online(query: str) -> str:
        """从 ArXiv + Semantic Scholar 搜索新论文。"""
        try:
            from .tools.search_orchestrator import SearchOrchestrator
            orch = SearchOrchestrator(storage, username)
            result = orch.search(query, sources=["arxiv", "s2"])
            papers = result.get("papers", [])
            if not papers:
                return f"外部搜索 '{query}': 未找到结果"
            lines = [f"搜索 '{query}': 找到 {result.get('total_found', 0)} 篇, 精选 Top-{len(papers)}"]
            for i, p in enumerate(papers[:5], 1):
                lines.append(
                    f"[{i}] {p.get('title', '?')}\n"
                    f"    评分: {p.get('composite_score', 0):.0f}/100 引用: {p.get('citation_count', 0) or 0} 年份: {p.get('year', '')}\n"
                    f"    核心: {p.get('core_contribution', '')[:200]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"（外部搜索失败: {e}）"

    @tool
    def query_knowledge_base(query: str, limit: int = 5) -> str:
        """搜索本地论文知识库。"""
        try:
            from .rag.retrieval import HybridRetriever
            retriever = HybridRetriever(storage, username)
            papers = retriever.search_papers_expanded(query, limit=limit, enable_mqe=False, enable_hyde=False)
            if not papers:
                return "（本地KB无结果）"
            items = []
            seen = set()
            for p in papers[:limit]:
                title = p.get("title", "?")
                if title in seen:
                    continue
                seen.add(title)
                items.append(f"- {title}\n  核心: {p.get('core_claim', '')[:120]}")
            return "\n".join(items)
        except Exception as e:
            return f"（KB失败: {e}）"

    return [search_papers_online, query_knowledge_base]


def _check_agent_satisfied(result_text: str, papers_count: int, iteration: int) -> bool:
    """检查 Agent 是否对搜索结果满意。

    启发式判断: LLM 输出含"满意/足够/完成"且论文>=3 或 已达3轮。
    """
    if iteration >= 3:
        return True
    if papers_count >= 8:
        return True
    satisfied_keywords = ["满意", "足够", "完备", "充分", "不再需要", "覆盖完整", "信息充足", "sufficient"]
    if any(kw in result_text for kw in satisfied_keywords) and papers_count >= 3:
        return True
    return False


def _parse_search_result(text: str) -> list[dict]:
    """从工具返回文本中解析论文信息。"""
    import re
    papers = []
    pattern = re.compile(r'\[(\d+)\]\s+(.+?)\n\s+评分:\s*([\d.]+).*?引用:\s*(\d+).*?年份:\s*(\d+)', re.DOTALL)
    for m in pattern.finditer(str(text)):
        papers.append({
            "index": int(m.group(1)),
            "title": m.group(2).strip(),
            "composite_score": float(m.group(3)),
            "citation_count": int(m.group(4)),
            "year": m.group(5),
        })
    return papers


# ============================================================
# 节点 ③ — synthesize: 综合撰写（按 workflow_type 分流）
# ============================================================

def node_synthesize(state: ResearchState) -> dict[str, Any]:
    """Agent 节点: 根据 workflow_type 产出综述/回答/报告。"""
    wf_type = state.get("workflow_type", "review")
    topic = state.get("topic", "")
    papers = state.get("papers_found", [])

    # 构建论文材料
    papers_material = _fmt_papers_for_synthesis(papers)

    # 三种路径各自的系统提示词
    system_map = {
        "review": SYNTHESIZE_REVIEW_SYSTEM,
        "research": SYNTHESIZE_ANSWER_SYSTEM,
        "progress_report": SYNTHESIZE_REPORT_SYSTEM,
    }
    system = system_map.get(wf_type, SYNTHESIZE_REVIEW_SYSTEM)

    type_hints = {
        "review": "请撰写一份正式文献综述（2500-4000字），含摘要/引言/研究脉络/方法对比/趋势/结论。末尾列出参考文献。",
        "research": "请基于论文材料深度回答用户问题。如果涉及后续方向，给出2-3个具体建议。",
        "progress_report": "请综合文献库和用户进展生成评估报告，含文献覆盖度/实验进展/知识缺口/文献对照/下一步建议。",
    }

    llm = get_llm(temperature=0.5, max_tokens=4096)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"研究主题: {topic}\n\n论文材料:\n{papers_material}\n\n{type_hints.get(wf_type, '')}"),
    ]
    response = llm.invoke(messages)
    output = str(response.content)

    # 提取引用
    cited = _extract_cited(papers, output)

    return {
        "final_output": output,
        "cited_papers": cited,
        "current_stage": "synthesize",
    }


def _fmt_papers_for_synthesis(papers: list[dict]) -> str:
    """格式化论文列表供 synthesize 节点使用。"""
    if not papers:
        return "（无搜索结果）"
    lines = []
    for i, p in enumerate(papers[:15], 1):
        title = p.get("title", "?")
        lines.append(
            f"[{i}] {title}\n"
            f"    评分: {p.get('composite_score', '?')} 引用: {p.get('citation_count', 0) or '?'} "
            f"年份: {p.get('year', '?')}\n"
            f"    核心: {p.get('core_contribution', p.get('core_claim', ''))[:200]}\n"
            f"    摘要: {(p.get('abstract', '') or '')[:300]}"
        )
    return "\n\n".join(lines)


def _extract_cited(papers: list[dict], output: str) -> list[dict]:
    """按引文标记提取论文。"""
    cited = []
    for i, p in enumerate(papers[:15]):
        title = p.get("title", "")[:30]
        if f"[{i+1}]" in output or title and title[:10] in output:
            cited.append(p)
    return cited


# ============================================================
# 节点 ④ — user_review: 人机协同审查
# ============================================================

def node_user_review(state: ResearchState) -> dict[str, Any]:
    """中断节点: 展示结果，等待用户确认或修改意见。

    LangGraph interrupt() 挂起执行，用户输入后继续。
    """
    output = state.get("final_output", "")
    cited = state.get("cited_papers", [])

    preview = output[:500] + ("..." if len(output) > 500 else "")

    prompt = (
        f"\n{'='*60}\n"
        f"文献综述已完成 ({len(output)}字, {len(cited)}篇引用)\n"
        f"{'='*60}\n"
        f"{preview}\n\n"
        f"[确认] 完成  [修改] 输入修改意见  [补充] 追加内容\n"
    )

    user_input = interrupt(prompt)

    if not user_input or user_input.strip().lower() in ("确认", "ok", "yes", "好", "可以"):
        return {
            "output_approved": True,
            "current_stage": "user_review",
        }

    # 用户有修改意见 → 追加到 final_output
    feedback = user_input.strip()
    return {
        "user_feedback": feedback,
        "output_approved": False,
        "current_stage": "user_review",
    }


# ============================================================
# 路由逻辑
# ============================================================

def route_after_understand(state: ResearchState) -> Literal["research", "__end__"]:
    """understand 后进入 research。"""
    return "research"


def route_after_research(state: ResearchState) -> Literal["synthesize", "research"]:
    """Agent 不满意 → 继续搜索；满意 → 进入综合。"""
    if state.get("agent_satisfied", False):
        return "synthesize"
    if state.get("search_iterations", 0) >= 3:
        return "synthesize"  # 最多3轮
    return "research"


def route_after_synthesize(state: ResearchState) -> Literal["user_review", "__end__"]:
    """review/report 走人审；research 直接结束。"""
    wf_type = state.get("workflow_type", "review")
    if wf_type == "research":
        return END
    return "user_review"


def route_after_user_review(state: ResearchState) -> Literal["synthesize", "__end__"]:
    """用户有修改意见 → 回 synthesize 调整；确认 → 结束。"""
    if state.get("output_approved", False):
        return END
    # 有修改意见: 把 feedback 追加到 topic 中 → 回 synthesize 重写
    feedback = state.get("user_feedback", "")
    if feedback:
        # 修改 topic 以包含反馈（简化处理）
        old_topic = state.get("topic", "")
        state["topic"] = f"{old_topic}（用户修改意见: {feedback}）"
    return "synthesize"


# ============================================================
# 构建 Graph
# ============================================================

def build_research_graph(
    checkpointer_path: str = "./data/checkpoints.db",
) -> CompiledStateGraph:
    """构建 Agent 驱动的科研工作流图。

    4 节点 / 3 路径 / research 节点可迭代。

    Args:
        checkpointer_path: SqliteSaver 数据库路径

    Returns:
        CompiledStateGraph（可直接 graph.stream() 调用）
    """
    os.makedirs(os.path.dirname(checkpointer_path) or ".", exist_ok=True)
    conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    workflow = StateGraph(ResearchState)

    # 4 个 Agent 节点
    workflow.add_node("understand", node_understand)
    workflow.add_node("research", node_research)
    workflow.add_node("synthesize", node_synthesize)
    workflow.add_node("user_review", node_user_review)

    # 入口
    workflow.set_entry_point("understand")

    # 边和条件路由
    workflow.add_edge("understand", "research")
    workflow.add_conditional_edges("research", route_after_research, {
        "synthesize": "synthesize",
        "research": "research",
    })
    workflow.add_conditional_edges("synthesize", route_after_synthesize, {
        "user_review": "user_review",
        END: END,
    })
    workflow.add_conditional_edges("user_review", route_after_user_review, {
        "synthesize": "synthesize",
        END: END,
    })

    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# 辅助函数
# ============================================================

def _get_thread_id() -> str:
    """从 LangGraph config 中获取 thread_id。"""
    try:
        config = get_config()
        return config.get("configurable", {}).get("thread_id", "")
    except Exception:
        return ""


def _exec_tool(tc: dict, tools: list) -> str:
    """执行工具调用，返回结果文本。"""
    name = tc.get("name", "")
    args = tc.get("args", {})
    for t in tools:
        if t.name == name:
            try:
                result = t.invoke(args)
                return str(result) if result else "（空结果）"
            except Exception as e:
                logger.warning("工具 %s 执行失败: %s", name, e)
                return f"（执行失败: {e}）"
    return f"（未知工具: {name}）"
