"""用户进展记录工具 — 记录 + 追踪闭环 + 搜索建议。

特性:
  - 进展记录后自动生成搜索/实验建议
  - 进展嵌入到 Qdrant（语义可召回）
  - 与计划联动更新
  - 进展时间线查看
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..utils import get_llm, extract_json_from_llm_response

PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

SUGGESTION_PROMPT = """基于用户的最新研究进展，生成下一步建议。

研究主题: {topic}
当前计划阶段: {plan_phase}

进展内容:
标题: {title}
类型: {entry_type}
内容: {content}
结果: {results}
洞察: {insights}

请以 JSON 格式输出:
{{
  "knowledge_gaps": ["识别到的知识缺口"],
  "suggested_searches": ["建议搜索的关键词（英文，2-3条）"],
  "suggested_experiments": ["建议的实验/验证步骤（1-2条）"],
  "plan_update_note": "对研究计划的影响说明（一句话）"
}}

只返回 JSON。"""


class ProgressEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    entry_type: str = "other"
    title: str = ""
    content: str = ""
    results: Optional[str] = None
    insights: Optional[str] = None
    next_actions: Optional[str] = None
    related_papers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


def _extract_metrics(content: str, results: str | None) -> dict:
    """从进展内容中提取定量指标。"""
    llm = get_llm(temperature=0.0, max_tokens=1024)
    prompt = f"""从研究进展描述中提取定量指标。

进展内容: {content}
实验结果: {results or '无'}

提取为 JSON:
{{"metric_name": numeric_value, ...}}

例如: {{"PLQY_before": 60, "PLQY_after": 85, "PLQY_24h": 70}}

只返回 JSON，数值用数字类型。"""

    try:
        response = llm.invoke([
            SystemMessage(content="只返回 JSON。"),
            HumanMessage(content=prompt),
        ])
        return extract_json_from_llm_response(str(response.content))
    except Exception:
        return {}


def _generate_suggestions(topic: str, title: str, entry_type: str,
                          content: str, results: str | None,
                          insights: str | None,
                          plan_phase: str = "") -> dict:
    """基于进展生成下一步建议。"""
    llm = get_llm(temperature=0.3, max_tokens=1024)

    try:
        response = llm.invoke([
            SystemMessage(content="只返回 JSON，建议用英文关键词。"),
            HumanMessage(content=SUGGESTION_PROMPT.format(
                topic=topic, title=title, entry_type=entry_type,
                content=content, results=results or "无",
                insights=insights or "无",
                plan_phase=plan_phase or "未知",
            )),
        ])
        return extract_json_from_llm_response(str(response.content))
    except (json.JSONDecodeError, AttributeError):
        return {
            "knowledge_gaps": [],
            "suggested_searches": [],
            "suggested_experiments": [],
            "plan_update_note": "",
        }


# === LangChain tools（供 graph 使用）===

@tool
def record_user_progress(
    topic: str,
    entry_type: str,
    title: str,
    content: str,
    results: Optional[str] = None,
    insights: Optional[str] = None,
    next_actions: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """记录用户自己的研究进展（实验、阅读、想法、结果等）。

    参数:
        topic: 研究主题
        entry_type: experiment / reading / idea / result / meeting / other
        title: 进展标题
        content: 详细描述
        results: 实验结果（数值、图表描述等）
        insights: 获得的洞察
        next_actions: 下一步计划
        tags: 逗号分隔标签
    """
    if not topic or not title:
        return "错误：研究主题和进展标题不能为空。"

    valid_types = ("experiment", "reading", "idea", "result", "meeting", "other")
    if entry_type not in valid_types:
        entry_type = "other"

    # 提取定量指标
    metrics = _extract_metrics(content, results)

    entry = ProgressEntry(
        entry_type=entry_type,
        title=title.strip(),
        content=content.strip(),
        results=results.strip() if results else None,
        insights=insights.strip() if insights else None,
        next_actions=next_actions.strip() if next_actions else None,
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        metrics=metrics,
    )

    # 保存到 JSON（传统备份）
    topic_slug = topic.strip().lower().replace(" ", "_")[:50]
    filepath = os.path.join(PROGRESS_DIR, f"{topic_slug}.json")
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.append(entry.model_dump())
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    # 生成建议
    suggestions = _generate_suggestions(
        topic, title, entry_type, content, results, insights,
    )

    type_labels = {
        "experiment": "实验", "reading": "文献阅读",
        "idea": "新想法", "result": "阶段性结果",
        "meeting": "讨论会", "other": "其他",
    }

    result = {
        "status": "ok",
        "message": f"已记录「{type_labels.get(entry_type, entry_type)}」进展: {entry.title}",
        "entry": entry.model_dump(),
        "suggestions": suggestions,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_progress_summary(topic: str) -> str:
    """查看某个研究主题的所有用户进展记录摘要。

    参数:
        topic: 研究主题

    返回:
        JSON 格式的进展记录列表，按时间倒序排列。
    """
    topic_slug = topic.strip().lower().replace(" ", "_")[:50]
    filepath = os.path.join(PROGRESS_DIR, f"{topic_slug}.json")

    if not os.path.exists(filepath):
        return json.dumps({
            "status": "ok", "topic": topic, "total_entries": 0,
            "message": f"「{topic}」暂无用户进展记录。",
        }, ensure_ascii=False, indent=2)

    with open(filepath, "r", encoding="utf-8") as f:
        entries = json.load(f)

    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    type_counts = {}
    for e in entries:
        t = e.get("entry_type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    # 时间线文本
    timeline = []
    for e in entries:
        ts = e.get("timestamp", "")[:10]
        t = e.get("entry_type", "other")
        emoji = {"experiment": "🔬", "reading": "📖", "idea": "💡",
                 "result": "📊", "meeting": "🤝"}.get(t, "📝")
        timeline.append(f"{ts} {emoji} {e.get('title', '')}")

    return json.dumps({
        "status": "ok",
        "topic": topic,
        "total_entries": len(entries),
        "type_breakdown": type_counts,
        "timeline": timeline,
        "latest_entries": entries[:10],
    }, ensure_ascii=False, indent=2)
