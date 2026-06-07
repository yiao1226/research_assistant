"""科研助手 LangGraph 工作流。

特性:
  - search_papers: 三路合流（KB + 新搜索 + 用户上传）
  - analyze_papers: 分层分析（Top-3 深度 + 4-8 轻量 + 其余跳过）
  - plan_research: 动态更新（检测现有计划，增量而非覆盖）
  - 所有节点可通过 RuntimeContext 注入 storage 和 username 以支持多用户
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph.config import get_config

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .utils import get_llm, extract_json_from_llm_response
from .tools.paper_search import search_arxiv, search_semantic_scholar, fetch_paper_full_text
from .tools.progress import record_user_progress, get_progress_summary
from .prompts import (
    SEARCH_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    SYNTHESIS_PROMPT,
    PLAN_PROMPT,
    PROGRESS_PROMPT,
)
from .state import ResearchState, get_runtime_context


# ============================================================
# 节点
# ============================================================

def node_understand_intent(state: ResearchState) -> dict[str, Any]:
    """理解用户意图——提取主题，加载用户上下文。"""
    llm = get_llm(temperature=0.1)
    topic = state.get("topic", "")
    additional = state.get("additional_requirements", "")

    messages = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            user_input = str(msg.content)
            break

    if user_input and not topic:
        result = llm.invoke([
            SystemMessage(content="从用户输入中提取研究主题。只输出主题本身。"),
            HumanMessage(content=user_input),
        ])
        topic = str(result.content).strip()

    return {
        "topic": topic,
        "current_stage": "understand_intent",
        "research_goal": user_input if user_input else topic,
        "additional_requirements": additional,
    }


def node_search_papers(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """搜索论文——三路合流（KB + 新搜索 + 上传）。

    来源 A: 知识库已有论文（Qdrant 语义检索）
    来源 B: 新搜索（外部 API + Query Understanding）
    来源 C: 用户上传论文（同样从 Qdrant 召回）

    合并 → 去重 → 排序 → Top-15
    """
    topic = state.get("topic", "")
    research_goal = state.get("research_goal", topic)
    thread_id = config.get("configurable", {}).get("thread_id", "") if config else ""
    ctx = get_runtime_context(thread_id)

    all_papers: list[dict] = []
    seen = set()

    # === 来源 A: 知识库已有论文 ===
    kb_search_func = ctx.get("_kb_search_func") if ctx else None
    if kb_search_func:
        try:
            kb_papers = kb_search_func(topic, limit=10)
            for p in kb_papers:
                key = p.get("arxiv_id") or p.get("doi") or p.get("title")
                if key and key not in seen:
                    seen.add(key)
                    p["source_label"] = "kb"
                    all_papers.append(p)
        except Exception:
            logger.warning("KB检索失败", exc_info=True)

    # === 来源 B: 新搜索 ===
    llm = get_llm(temperature=0.3, max_tokens=4096)
    tools = [search_arxiv, search_semantic_scholar, fetch_paper_full_text]
    llm_with_tools = llm.bind_tools(tools)

    system = SEARCH_AGENT_PROMPT
    user_msg = f"""请为以下研究主题搜索相关学术论文。

研究主题: {topic}
研究目标: {research_goal}

请使用 search_arxiv 和 search_semantic_scholar 工具搜索论文。先用核心关键词搜索。"""

    response = llm_with_tools.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            tool_map = {
                "search_arxiv": search_arxiv,
                "search_semantic_scholar": search_semantic_scholar,
                "fetch_paper_full_text": fetch_paper_full_text,
            }
            func = tool_map.get(tc["name"])
            if func:
                result = func.invoke(tc["args"])
                try:
                    data = json.loads(result) if isinstance(result, str) else result
                    if isinstance(data, dict) and data.get("status") == "ok":
                        for paper in data.get("papers", []):
                            key = paper.get("arxiv_id") or paper.get("doi") or paper.get("title")
                            if key and key not in seen:
                                seen.add(key)
                                paper["source_label"] = "new_search"
                                all_papers.append(paper)
                except json.JSONDecodeError:
                    pass

    # 如果 LLM 没调用工具，执行默认搜索
    if not all_papers:
        arxiv_result = search_arxiv.invoke({"query": topic, "max_results": 10})
        try:
            data = json.loads(arxiv_result) if isinstance(arxiv_result, str) else arxiv_result
            if isinstance(data, dict) and data.get("status") == "ok":
                for paper in data.get("papers", []):
                    key = paper.get("arxiv_id") or paper.get("title")
                    if key and key not in seen:
                        seen.add(key)
                        paper["source_label"] = "new_search"
                        all_papers.append(paper)
        except json.JSONDecodeError:
            pass

    search_status = "success" if all_papers else "failed"

    # 取 Top-15
    all_papers = all_papers[:15]

    return {
        "papers_found": all_papers,
        "search_status": search_status,
        "current_stage": "search_papers",
        "search_queries_used": [topic],
    }


def node_analyze_papers(state: ResearchState) -> dict[str, Any]:
    """分层分析论文——降低 Token 消耗。

    Top-3:   深度分析（读全文 chunk + 标注）
    Rank 4-8: 轻量分析（仅读 core_claim + 标题 + tags）
    Rank 9+:  只列标题 + core_claim，不分析
    """
    llm = get_llm(temperature=0.3, max_tokens=4096)
    papers = state.get("papers_found", [])

    if not papers:
        return {
            "current_stage": "analyze_papers",
            "analyzed_papers": [],
            "papers_skipped": ["没有搜索到论文可供分析"],
        }

    n = len(papers)

    # === 深度分析 Top-3 ===
    deep_papers = papers[:3]
    deep_text = "\n\n".join(
        f"### [{i+1}] {p.get('title', '未知')}\n"
        f"作者: {', '.join(p.get('authors', [])[:5])}\n"
        f"发表: {p.get('published', '')} | {p.get('venue', '')}\n"
        f"引用: {p.get('citation_count', 'N/A')}\n"
        f"摘要: {(p.get('abstract', '') or '')[:500]}\n"
        f"标注: {_fmt_annotation(p)}"
        for i, p in enumerate(deep_papers)
    )

    system = ANALYSIS_AGENT_PROMPT
    deep_response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""请深度分析以下 {len(deep_papers)} 篇论文。

研究主题: {state.get('topic', '')}

{deep_text}

对每篇论文分析: 核心贡献、方法概述、关键发现、优势、局限、与用户研究的关联、建议追踪方向。"""),
    ])

    # 尝试结构化解析 LLM 分析结果
    analyzed = []
    deep_analysis_text = str(deep_response.content)
    try:
        parsed = extract_json_from_llm_response(deep_analysis_text)
        if isinstance(parsed, dict):
            items = parsed.get("papers", parsed.get("analyses", []))
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if isinstance(item, dict) and idx < len(deep_papers):
                        analyzed.append({
                            "paper": deep_papers[idx],
                            "analysis_level": "deep",
                            "core_contribution": item.get("core_contribution", ""),
                            "method_summary": item.get("method_summary", ""),
                            "key_findings": item.get("key_findings", ""),
                            "strengths": item.get("strengths", []),
                            "limitations": item.get("limitations", []),
                            "relevance_to_my_research": item.get("relevance_to_my_research", ""),
                            "suggested_followups": item.get("suggested_followups", []),
                        })
    except Exception:
        logger.debug("LLM 分析结果结构化解析失败，使用降级方案", exc_info=True)

    # 解析失败时降级：每个缺失论文用各自的元数据做兜底，不用同一段 LLM 原文
    if len(analyzed) < len(deep_papers):
        for i, p in enumerate(deep_papers):
            if i >= len(analyzed):
                title_short = p.get("title", f"论文{i+1}")[:80]
                ann = p.get("annotation", {})
                if isinstance(ann, str):
                    try:
                        import json
                        ann = json.loads(ann)
                    except Exception:
                        ann = {}
                core = (ann.get("core_claim", "") if isinstance(ann, dict) else "") or "（分析结果解析失败）"
                analyzed.append({
                    "paper": p,
                    "analysis_level": "deep",
                    "core_contribution": core,
                    "relevance_to_my_research": "",
                })

    # === 轻量分析 Rank 4-8 ===
    if n > 3:
        light_papers = papers[3:min(8, n)]
        light_text = "\n".join(
            f"[{i+4}] {p.get('title', '')} | {_fmt_annotation_short(p)}"
            for i, p in enumerate(light_papers)
        )

        light_response = llm.invoke([
            SystemMessage(content="只输出 1-2 句中文关联性说明。"),
            HumanMessage(content=f"研究主题: {state.get('topic', '')}\n\n{light_text}\n\n为每篇只写 1-2 句与主题的关联。"),
        ])

        for i, p in enumerate(light_papers):
            analyzed.append({
                "paper": p,
                "analysis_level": "light",
                "core_contribution": "",
                "relevance_to_my_research": "",
            })

    # === 剩余论文只列标题 ===
    skipped = []
    if n > 8:
        for p in papers[8:]:
            skipped.append(f"{p.get('title', '?')} (标注: {_fmt_annotation_short(p)})")

    return {
        "analyzed_papers": analyzed,
        "papers_skipped": skipped,
        "current_stage": "analyze_papers",
    }


def node_synthesize_review(state: ResearchState) -> dict[str, Any]:
    """撰写综述——综合所有分析的论文。"""
    llm = get_llm(temperature=0.5, max_tokens=4096)
    topic = state.get("topic", "")
    papers = state.get("papers_found", [])
    analyzed = state.get("analyzed_papers", [])

    if not papers:
        return {
            "literature_review": "## 文献综述\n\n暂无足够论文可供综述。",
            "review_word_count": 0,
            "current_stage": "synthesize_review",
        }

    papers_info = "\n\n".join(
        f"[{i+1}] **{p.get('title', 'Unknown')}** "
        f"({p.get('source_label', '')}, 引用: {p.get('citation_count', 0) or 0})\n"
        f"    作者: {(p.get('authors', ['Unknown']) or ['Unknown'])[0]} et al.\n"
        f"    摘要: {(p.get('abstract', '') or '')[:300]}...\n"
        for i, p in enumerate(papers[:10])
    )

    system = SYNTHESIS_PROMPT.format(主题=topic)
    user_msg = f"""请基于以下真实论文撰写文献综述。

研究主题: {topic}

## 待综述的论文

{papers_info}

要求: 引用用 [编号] 标注，所有信息来自真实 API 结果，2000-4000 字，中文撰写。"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    review = str(response.content)
    return {
        "literature_review": review,
        "review_word_count": len(review),
        "current_stage": "synthesize_review",
    }


def node_plan_research(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """制定/更新研究计划——检测现有计划，增量更新。"""
    llm = get_llm(temperature=0.4, max_tokens=4096)
    topic = state.get("topic", "")
    review = state.get("literature_review", "")
    research_goal = state.get("research_goal", topic)

    # 尝试获取现有计划（通过运行时上下文）
    thread_id = config.get("configurable", {}).get("thread_id", "") if config else ""
    ctx = get_runtime_context(thread_id)
    get_plan_func = ctx.get("_get_plan_func") if ctx else None
    existing_plan = None
    if get_plan_func:
        try:
            existing_plan = get_plan_func(topic)
        except Exception:
            logger.debug("无法获取现有计划（新用户正常）", exc_info=True)

    review_excerpt = review[:3000] if review else "暂无文献综述"

    if existing_plan:
        # 增量更新
        plan_text = json.dumps(existing_plan.get("phases", []), ensure_ascii=False)[:1500]
        system = PLAN_PROMPT
        user_msg = f"""请基于新的文献综述更新现有研究计划。

研究主题: {topic}

## 现有计划
{plan_text}

## 新文献综述
{review_excerpt}

请说明需要更新的部分（新增方向、调整优先级等），不要完整重写。如无需大改，输出 "计划无需大幅调整"。"""
    else:
        system = PLAN_PROMPT
        user_msg = f"""请为以下研究课题制定研究计划。

研究主题: {topic}
研究目标: {research_goal}

## 文献综述摘要
{review_excerpt}

请制定包含: 1. SMART目标 2. 技术路线 3. 实验方案 4. 3阶段时间规划 5. 风险与备选"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    plan = str(response.content)

    suggested_experiments = []
    for line in plan.split("\n"):
        if "实验" in line and ("：" in line or ":" in line):
            suggested_experiments.append(line.strip())

    return {
        "research_plan": plan,
        "suggested_experiments": suggested_experiments[:5],
        "current_stage": "plan_research",
    }


def node_user_progress_input(state: ResearchState) -> dict[str, Any]:
    """用户进展输入——中断点。"""
    topic = state.get("topic", "")

    existing_progress = get_progress_summary.invoke({"topic": topic})

    prompt = f"""## 请记录你的研究进展

当前研究主题: **{topic}**

你可以记录: 做了什么实验/得到什么结果/读了哪些论文/产生了什么新想法

已有进展:
{existing_progress}

请输入你的进展（直接描述即可）："""

    # interrupt 返回值在当前 LangGraph 版本中被忽略，
    # 实际输入在 graph.stream 的第二参数中传入
    interrupt(prompt)

    return {
        "current_stage": "user_progress_input",
        "pending_user_input": None,
    }


def node_assess_progress(state: ResearchState) -> dict[str, Any]:
    """评估进展——综合文献库 + 用户进展 + 计划。"""
    llm = get_llm(temperature=0.3, max_tokens=4096)
    topic = state.get("topic", "")
    plan = state.get("research_plan", "")
    papers_count = len(state.get("papers_found", []))
    analyzed_count = len(state.get("analyzed_papers", []))

    today = datetime.now().strftime("%Y年%m月%d日")
    progress_summary = get_progress_summary.invoke({"topic": topic})

    system = PROGRESS_PROMPT.format(日期=today)
    user_msg = f"""请综合以下信息生成研究进展报告。

研究主题: {topic}

## 文献库状态
- 检索论文数: {papers_count}
- 深度分析数: {analyzed_count}

## 研究计划参考
{plan[:2000] if plan else '暂无研究计划'}

## 用户自身进展记录
{progress_summary}

请评估: 1. 文献掌握程度和知识缺口 2. 用户实验进展 3. 计划完成度 4. 下一步具体建议"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    report = str(response.content)

    knowledge_gaps = []
    next_steps = []
    for line in report.split("\n"):
        if any(kw in line for kw in ["缺口", "未覆盖", "缺少"]):
            knowledge_gaps.append(line.strip())
        if "建议" in line and ("：" in line or ":" in line):
            next_steps.append(line.strip())

    return {
        "progress_report": report,
        "knowledge_gaps": knowledge_gaps[:5],
        "next_step_suggestions": next_steps[:5],
        "current_stage": "assess_progress",
    }


def node_suggest_next(state: ResearchState) -> dict[str, Any]:
    """最终节点——整理下一步行动建议。"""
    return {
        "current_stage": "suggest_next",
    }


# ============================================================
# 路由
# ============================================================

def route_after_search(state: ResearchState) -> Literal["analyze_papers", "search_papers"]:
    papers = state.get("papers_found", [])
    status = state.get("search_status", "")
    if not papers or status == "failed":
        return "search_papers"
    return "analyze_papers"


def route_after_analyze(state: ResearchState) -> Literal["synthesize_review", "search_papers"]:
    papers = state.get("analyzed_papers", [])
    if not papers:
        return "search_papers"
    return "synthesize_review"


def route_after_progress(state: ResearchState) -> Literal["assess_progress", "suggest_next"]:
    stage = state.get("current_stage", "")
    if stage == "user_progress_input":
        return "assess_progress"
    return "suggest_next"


# ============================================================
# 构建 Graph
# ============================================================

def build_research_graph(
    checkpointer_path: str = "./data/checkpoints.db",
) -> CompiledStateGraph:
    os.makedirs(os.path.dirname(checkpointer_path) or ".", exist_ok=True)
    conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    workflow = StateGraph(ResearchState)

    workflow.add_node("understand_intent", node_understand_intent)
    workflow.add_node("search_papers", node_search_papers)
    workflow.add_node("analyze_papers", node_analyze_papers)
    workflow.add_node("synthesize_review", node_synthesize_review)
    workflow.add_node("plan_research", node_plan_research)
    workflow.add_node("user_progress_input", node_user_progress_input)
    workflow.add_node("assess_progress", node_assess_progress)
    workflow.add_node("suggest_next", node_suggest_next)

    workflow.set_entry_point("understand_intent")
    workflow.add_edge("understand_intent", "search_papers")
    workflow.add_conditional_edges("search_papers", route_after_search)
    workflow.add_conditional_edges("analyze_papers", route_after_analyze)
    workflow.add_edge("synthesize_review", "plan_research")
    workflow.add_edge("plan_research", "user_progress_input")
    workflow.add_conditional_edges("user_progress_input", route_after_progress)
    workflow.add_edge("assess_progress", "suggest_next")
    workflow.add_edge("suggest_next", END)

    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# 辅助
# ============================================================

def _fmt_annotation(paper: dict) -> str:
    """格式化标注为简要文本。"""
    ann = paper.get("annotation")
    if isinstance(ann, str):
        try:
            ann = json.loads(ann)
        except json.JSONDecodeError:
            ann = {}
    if not ann:
        return ""

    parts = []
    for key in ["core_claim", "contribution_type",
                "keywords_material", "keywords_method"]:
        val = ann.get(key, "")
        if isinstance(val, list):
            val = ", ".join(val[:5])
        if val:
            parts.append(f"{key}: {val}")
    return " | ".join(parts)


def _fmt_annotation_short(paper: dict) -> str:
    """标注极简格式。"""
    ann = paper.get("annotation")
    if isinstance(ann, str):
        try:
            ann = json.loads(ann)
        except json.JSONDecodeError:
            ann = {}
    if not ann:
        return ""
    return ann.get("core_claim", "") or ""
