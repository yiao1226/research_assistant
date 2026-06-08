"""科研助手 LangGraph 工作流 — Agent 驱动 + Hook。

4 节点 / 3 路径 / research 可迭代 / Hook 可挂载

Hook 事件:
  node_start(name, state) / node_end(name, state)
  research_iteration(n, papers_count, satisfied)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph.config import get_config

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..utils import get_llm
from ..state import ResearchState, get_runtime_context
from ..tools import make_all_agent_tools
from ..tools.kb import make_kb_tools
from ..tools.history import make_history_tool
from ..agent.qa import _exec_tool

# ============================================================
# Graph Hook 系统
# ============================================================

GRAPH_HOOKS: dict[str, list[Callable]] = {
    "node_start": [],  # (node_name, state) → None
    "node_end": [],    # (node_name, state) → None
    "research_loop": [],  # (iteration, papers_count, satisfied) → None
}


def register_graph_hook(event: str, callback: Callable):
    GRAPH_HOOKS[event].append(callback)


def _trigger(event: str, *args):
    for cb in GRAPH_HOOKS[event]:
        cb(*args)

# ============================================================
# System Prompt 模板（动态注入工具描述）
# ============================================================

NODE_PROMPTS = {
    "understand": """你是科研策略规划专家。在搜索之前先盘点已有知识基础。

## 可用工具
{tools}

## 你需要产出
1. 已有基础: KB有什么论文、进展记录了哪些实验、历史讨论过什么
2. 搜索计划: 关键词(中英各2-3)、重点方向、搜索策略""",

    "research": """你是学术文献搜索专家。搜索论文并评估质量。

## 可用工具
{tools}

## 策略
1. 核心关键词搜索 → 评估覆盖度和质量
2. 不足就换角度重搜(如加method/fabrication关键词)
3. 满意则输出评估: 总篇数、方向覆盖、质量评价

## 满意标准
论文>=5且方向覆盖完整、有足够方法细节。最多建议3轮搜索。""",

    "synthesize_review": """你是学术综述专家。撰写正式文献综述。
结构: 摘要/引言/研究脉络/方法对比(表格)/趋势/结论+参考文献
要求: 引用[编号], 中文, 2500-4000字""",

    "synthesize_answer": """你是科研问答专家。基于搜索分析深度回答问题。
要求: 有据可查, 引用[论文], 含具体数值""",

    "synthesize_report": """你是科研进展评估专家。综合文献库和进展记录生成报告。
结构: 文献覆盖度/实验进展/知识缺口/文献对照/下一步建议""",
}


def build_node_prompt(tools: list, node_type: str) -> str:
    """从工具定义动态生成节点 System Prompt。

    工具列表从 @tool 的 description 自动提取，不硬编码。
    synthesize 节点不绑工具，不需要 {tools} 占位。
    """
    template = NODE_PROMPTS.get(node_type, NODE_PROMPTS["understand"])

    if "{tools}" in template:
        tool_lines = [f"- **{t.name}**: {t.description.split(chr(10))[0]}" for t in tools]
        return template.format(tools="\n".join(tool_lines))
    return template

# ============================================================
# 节点 ① — understand
# ============================================================

def node_understand(state: ResearchState) -> dict[str, Any]:
    """Agent 盘点现状 + 制定搜索策略。"""
    topic = state.get("topic", "")
    wf_type = state.get("workflow_type", "review")
    ctx = get_runtime_context(_get_thread_id())
    storage = ctx.get("_storage")
    username = ctx.get("_username", "default")

    llm = get_llm(temperature=0.2, max_tokens=600)
    tools = make_kb_tools(storage, username) + [make_history_tool(storage, username)]
    system = build_node_prompt(tools, "understand")
    llm_with_tools = llm.bind_tools(tools)

    wf_labels = {"review": "文献综述", "research": "深度研究", "progress_report": "进展评估"}
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"主题: {topic}\n类型: {wf_labels.get(wf_type, wf_type)}\n请盘点已有基础，制定搜索计划。"),
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

    _trigger("node_end", "understand", {"search_plan": plan_text})
    return {"search_plan": plan_text, "current_stage": "understand"}


# ============================================================
# 节点 ② — research: 搜索分析循环
# ============================================================

def node_research(state: ResearchState) -> dict[str, Any]:
    """Agent 迭代搜索 + 自评估。不满意换策略重搜，最多3轮。"""
    topic = state.get("topic", "")
    plan = state.get("search_plan", "")
    iteration = state.get("search_iterations", 0) + 1
    papers = list(state.get("papers_found", []))
    ctx = get_runtime_context(_get_thread_id())
    storage = ctx.get("_storage")
    username = ctx.get("_username", "default")

    llm = get_llm(temperature=0.3, max_tokens=1024)
    tools = make_all_agent_tools(storage, username)
    system = build_node_prompt(tools, "research")
    llm_with_tools = llm.bind_tools(tools)

    prev_summary = f"已有 {len(papers)} 篇" if papers else "尚无结果"
    prev_titles = "\n".join(f"- {p.get('title', '?')[:80]}" for p in papers[-10:]) if papers else ""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"主题: {topic}\n\n搜索计划: {plan}\n\n第{iteration}轮搜索\n{prev_summary}\n{prev_titles}\n\n请搜索论文并评估。"),
    ]

    result_text = ""
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            result_text = str(response.content)
            break
        messages.append(response)
        for tc in response.tool_calls:
            tr = _exec_tool(tc, tools)
            if tc["name"] == "search_papers_online":
                _merge_papers_from_text(papers, str(tr))
            messages.append(ToolMessage(content=str(tr), tool_call_id=tc.get("id", "")))

    satisfied = _check_satisfied(result_text, len(papers), iteration)
    _trigger("research_loop", iteration, len(papers), satisfied)

    return {
        "papers_found": papers,
        "search_iterations": iteration,
        "agent_satisfied": satisfied,
        "search_summary": result_text[:500],
        "current_stage": "research",
    }


def _merge_papers_from_text(papers: list, text: str):
    """从工具返回文本解析并合并论文（去重）。

    优先使用结构化正则匹配，失败时回退到宽松标题提取。
    """
    import re
    seen = {p.get("title", "") for p in papers}
    matched = False

    # 主路径：结构化正则匹配
    for m in re.finditer(r'\[(\d+)\]\s+(.+?)\n\s+评分:\s*([\d.]+).*?引用:\s*(\d+).*?年份:\s*(\d+)', str(text), re.DOTALL):
        title = m.group(2).strip()
        if title not in seen:
            seen.add(title)
            papers.append({
                "index": int(m.group(1)), "title": title,
                "composite_score": float(m.group(3)),
                "citation_count": int(m.group(4)), "year": m.group(5),
            })
            matched = True

    # 降级：宽松提取 `[N] 标题` 行（正则失败时兜底）
    if not matched:
        for m in re.finditer(r'\[(\d+)\]\s+(.+)', str(text)):
            title = m.group(2).strip()
            # 截取到第一个换行或评分前
            title = re.split(r'\n|评分:', title)[0].strip()
            if title and title not in seen and len(title) > 5:
                seen.add(title)
                papers.append({
                    "index": int(m.group(1)), "title": title,
                    "composite_score": 0.0,
                    "citation_count": 0, "year": "",
                })


def _check_satisfied(text: str, count: int, iteration: int) -> bool:
    """Agent 是否对结果满意。"""
    if iteration >= 3 or count >= 8:
        return True
    kw = ["满意", "足够", "完备", "充分", "覆盖完整", "信息充足", "sufficient"]
    if any(k in text for k in kw) and count >= 3:
        return True
    return False


# ============================================================
# 节点 ③ — synthesize
# ============================================================

def node_synthesize(state: ResearchState) -> dict[str, Any]:
    """按 workflow_type 产出综述/回答/报告。"""
    wf_type = state.get("workflow_type", "review")
    topic = state.get("topic", "")
    papers = state.get("papers_found", [])

    material = _fmt_papers(papers)
    node_map = {"review": "synthesize_review", "research": "synthesize_answer", "progress_report": "synthesize_report"}
    system = build_node_prompt([], node_map.get(wf_type, "synthesize_review"))

    hints = {
        "review": "撰写正式文献综述(2500-4000字), 含摘要/引言/脉络/对比表格/趋势/结论/参考文献。",
        "research": "深度回答问题，有引用，如需后续方向给出2-3个建议。",
        "progress_report": "生成评估报告，含文献覆盖/实验进展/知识缺口/文献对照/下一步建议。",
    }

    llm = get_llm(temperature=0.5, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"主题: {topic}\n\n论文材料:\n{material}\n\n{hints.get(wf_type, '')}"),
    ])
    output = str(response.content)

    cited = []
    for i, p in enumerate(papers[:15]):
        if f"[{i+1}]" in output:
            cited.append(p)

    return {"final_output": output, "cited_papers": cited, "current_stage": "synthesize"}


def _fmt_papers(papers: list[dict]) -> str:
    if not papers:
        return "（无搜索结果）"
    lines = []
    for i, p in enumerate(papers[:15], 1):
        lines.append(
            f"[{i}] {p.get('title', '?')}\n"
            f"    评分: {p.get('composite_score', '?')} 引用: {p.get('citation_count', 0) or '?'} "
            f"年份: {p.get('year', '?')}\n"
            f"    核心: {p.get('core_contribution', p.get('core_claim', ''))[:200]}\n"
            f"    摘要: {(p.get('abstract', '') or '')[:300]}"
        )
    return "\n\n".join(lines)


# ============================================================
# 节点 ④ — user_review
# ============================================================

def node_user_review(state: ResearchState) -> dict[str, Any]:
    """人机协同: 展示结果，等确认/修改。"""
    output = state.get("final_output", "")
    cited = state.get("cited_papers", [])

    preview = output[:500] + ("..." if len(output) > 500 else "")
    prompt = (
        f"\n{'='*60}\n"
        f"工作流完成 ({len(output)}字, {len(cited)}篇引用)\n"
        f"{'='*60}\n"
        f"{preview}\n\n"
        f"[确认]完成  [修改]输入意见  [补充]追加内容\n"
    )

    user_input = interrupt(prompt)

    if not user_input or user_input.strip().lower() in ("确认", "ok", "yes", "好", "可以"):
        return {"output_approved": True, "current_stage": "user_review"}

    return {"user_feedback": user_input.strip(), "output_approved": False, "current_stage": "user_review"}


# ============================================================
# 路由
# ============================================================

def route_after_research(state: ResearchState) -> Literal["synthesize", "research"]:
    if state.get("agent_satisfied", False) or state.get("search_iterations", 0) >= 3:
        return "synthesize"
    return "research"


def route_after_synthesize(state: ResearchState) -> Literal["user_review", "__end__"]:
    return END if state.get("workflow_type") == "research" else "user_review"


def route_after_user_review(state: ResearchState) -> Literal["synthesize", "__end__"]:
    if state.get("output_approved", False):
        return END
    feedback = state.get("user_feedback", "")
    if feedback:
        state["topic"] = f"{state.get('topic', '')}（修改意见: {feedback}）"
    return "synthesize"


# ============================================================
# 构建
# ============================================================

def build_research_graph(checkpointer_path: str = "./data/checkpoints.db") -> CompiledStateGraph:
    os.makedirs(os.path.dirname(checkpointer_path) or ".", exist_ok=True)
    conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    workflow = StateGraph(ResearchState)

    workflow.add_node("understand", node_understand)
    workflow.add_node("research", node_research)
    workflow.add_node("synthesize", node_synthesize)
    workflow.add_node("user_review", node_user_review)

    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "research")
    workflow.add_conditional_edges("research", route_after_research, {"synthesize": "synthesize", "research": "research"})
    workflow.add_conditional_edges("synthesize", route_after_synthesize, {"user_review": "user_review", END: END})
    workflow.add_conditional_edges("user_review", route_after_user_review, {"synthesize": "synthesize", END: END})

    return workflow.compile(checkpointer=checkpointer)


def _get_thread_id() -> str:
    try:
        return get_config().get("configurable", {}).get("thread_id", "")
    except Exception:
        return ""
