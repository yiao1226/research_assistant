"""研究计划管理工具 — 动态更新、进展联动、方向提示。

计划生命周期:
  1. review 工作流 plan_research 节点 → 初始生成
  2. 用户每次记录进展 → 触发进度更新
  3. 新论文入库 → 检测新方向 → 提示用户
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..utils import get_llm, extract_json_from_llm_response

logger = logging.getLogger(__name__)


PLAN_UPDATE_PROMPT = """你是科研计划管理专家。检测用户的进展是否影响现有研究计划。

现有研究计划:
{current_plan}

用户最新进展:
{recent_progress}

已入库论文方向:
{paper_directions}

请判断:
1. 进展属于计划的哪个阶段/任务？
2. 哪些任务已完成？完成度更新到多少？
3. 是否有新方向需要加入计划？

返回 JSON:
{{
  "phase_updates": [
    {{"phase_index": 0, "task_updates": [{{"task_index": 0, "status": "done"}}]}}
  ],
  "overall_progress": 45.0,
  "new_direction_alert": "新方向描述（如有）",
  "suggested_plan_adjustment": "建议调整（如有）"
}}

只返回 JSON。"""

NEW_DIRECTION_PROMPT = """你发现了可能与用户当前研究计划相关的新方向。

当前计划: {plan_summary}

新入库论文:
{new_papers}

这些论文是否揭示了与当前计划不同的新方向？如果有，请用一句话描述。
只输出一句话描述，如果没有明显新方向就输出 "无"。"""


class Task(BaseModel):
    title: str = ""
    status: str = "pending"  # pending / in_progress / done
    linked_progress: list[str] = Field(default_factory=list)
    linked_papers: list[str] = Field(default_factory=list)


class Phase(BaseModel):
    title: str = ""
    status: str = "pending"  # pending / in_progress / done
    tasks: list[Task] = Field(default_factory=list)
    expected_output: str = ""
    start_date: str = ""
    end_date: str = ""


def detect_new_directions(storage, new_papers: list[dict], topic: str) -> str | None:
    """检测新入库论文是否揭示了新方向。

    Args:
        storage: PerUserStorage 实例
        new_papers: 新入库的论文列表
        topic: 当前研究主题

    Returns:
        新方向描述，或 None
    """
    plan = storage.get_active_plan(topic)
    if not plan:
        return None

    llm = get_llm(temperature=0.1)

    plan_summary = json.dumps(plan.get("phases", []), ensure_ascii=False)[:1000]
    papers_text = "\n".join(
        f"- {p.get('title', '')}: {p.get('core_claim', '')}"
        for p in new_papers[:5]
    )

    try:
        response = llm.invoke([
            SystemMessage(content="只输出一句话，不输出其他内容。"),
            HumanMessage(content=NEW_DIRECTION_PROMPT.format(
                plan_summary=plan_summary,
                new_papers=papers_text[:2000],
            )),
        ])
        text = str(response.content).strip()
        if text and text != "无":
            return text
    except Exception:
        logger.debug("新方向检测失败", exc_info=True)
    return None


def update_plan_from_progress(storage, topic: str,
                              recent_progress: list[dict],
                              paper_directions: list[str]) -> dict | None:
    """基于用户进展更新研究计划。

    Returns:
        更新摘要 dict，或 None（无需更新）
    """
    plan = storage.get_active_plan(topic)
    if not plan:
        return None

    llm = get_llm(temperature=0.2)

    progress_text = "\n".join(
        f"- [{p.get('timestamp', '')}] {p.get('title', '')}: {p.get('content', '')[:100]}"
        for p in recent_progress[-5:]
    )

    try:
        response = llm.invoke([
            SystemMessage(content="你是科研计划管理专家。只返回 JSON。"),
            HumanMessage(content=PLAN_UPDATE_PROMPT.format(
                current_plan=json.dumps(plan.get("phases", []), ensure_ascii=False)[:2000],
                recent_progress=progress_text,
                paper_directions="、".join(paper_directions[:10]),
            )),
        ])
        result = extract_json_from_llm_response(str(response.content))
    except Exception:
        logger.warning("计划更新 LLM 解析失败", exc_info=True)
        return None

    # 应用更新
    phases = plan.get("phases", [])
    if isinstance(phases, str):
        phases = json.loads(phases)

    for pu in result.get("phase_updates", []):
        pi = pu.get("phase_index", -1)
        if 0 <= pi < len(phases):
            for tu in pu.get("task_updates", []):
                ti = tu.get("task_index", -1)
                if 0 <= ti < len(phases[pi].get("tasks", [])):
                    if isinstance(phases[pi]["tasks"][ti], dict):
                        phases[pi]["tasks"][ti]["status"] = tu.get("status", "pending")
                    else:
                        phases[pi]["tasks"][ti] = {"title": str(phases[pi]["tasks"][ti]),
                                                    "status": tu.get("status", "pending")}

    overall = result.get("overall_progress", plan.get("overall_progress", 0))

    # 写入 SQLite
    storage.update_plan_progress(
        plan["id"], phases,
        result.get("current_phase", plan.get("current_phase", 0)),
        overall,
    )

    return {
        "topic": topic,
        "overall_progress": overall,
        "phase_updates": result.get("phase_updates", []),
        "new_direction_alert": result.get("new_direction_alert"),
        "suggested_adjustment": result.get("suggested_plan_adjustment"),
    }
