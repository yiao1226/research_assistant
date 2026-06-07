"""结构化事实提取 — 参考 DeerFlow 的 MemoryUpdater + prompt 模块。

从对话中提取带置信度的结构化事实，支持:
  - 事实分类（preference / context / experiment_detail / correction 等）
  - 置信度评分（0.0-1.0）
  - 增量合并（同内容事实覆盖旧事实、低置信度过滤、超量裁剪）
  - 纠正信号触发优先修正
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from typing import Any

from ..utils import get_llm, extract_json_from_llm_response

logger = logging.getLogger(__name__)

# ── 提示词 ──

FACT_EXTRACTION_PROMPT = """从以下对话中提取关于用户的结构化信息。

你是科研助手的记忆模块。你的任务是阅读对话并提取:
1. 用户的科研相关事实（实验参数、方法偏好、材料选择、设备信息等）
2. 用户的偏好和习惯（喜欢什么格式、关注什么领域等）
3. 用户研究背景（做什么方向、有什么经验等）

## 对话内容
{conversation_text}

## 已有事实（供参考，避免重复）
{existing_facts_summary}

## 输出格式
只返回 JSON:
{{
  "userContext": {{
    "researchFocus": "用户当前的研究方向（1-2句话）",
    "methodPreference": "用户偏好的实验/计算方法（1句话，无明确信息则空字符串）",
    "expertiseLevel": "用户在该领域的经验水平（novice|intermediate|advanced|unknown）"
  }},
  "newFacts": [
    {{
      "content": "用户在800°C下进行CVD沉积实验",
      "category": "experiment_detail",
      "confidence": 0.95
    }}
  ],
  "factsToRemove": []
}}

## 分类体系
- preference: 用户偏好（格式、风格、工具等）
- context: 背景信息（工作单位、研究领域等）
- experiment_detail: 实验细节（参数、条件、设备）
- method_knowledge: 方法学知识（熟悉什么技术）
- correction: 纠正（用户指出AI错误后提取的正确信息）

## 置信度标准
- 0.95-1.0: 用户明确说出（"我用CVD方法"）
- 0.70-0.94: 从对话中合理推断
- 0.50-0.69: 间接暗示，可能正确
- <0.50: 不要输出（过滤掉）

## 规则
- 不要提取论文内容本身（论文属于知识库，不是用户信息）
- 不要提取文件路径、上传事件等临时信息
- 如果对话无新信息，返回空数组
- 标记应该删除的旧事实（用户纠正了、信息过时了）放入 factsToRemove

{correction_hint}

只返回 JSON。"""


# ── 公共函数 ──

def extract_facts_from_conversation(
    turns: list,
    existing_facts: list[dict] | None = None,
    *,
    correction_detected: bool = False,
    reinforcement_detected: bool = False,
) -> dict:
    """从对话轮次中提取结构化事实。

    Args:
        turns: QATurn 对象列表
        existing_facts: 已有的事实列表（用于去重和合并）
        correction_detected: 本轮是否检测到纠正信号
        reinforcement_detected: 本轮是否检测到认可信号

    Returns:
        {
            "userContext": {...},
            "newFacts": [{content, category, confidence, ...}],
            "factsToRemove": [fact_id, ...],
        }
    """
    if not turns:
        return {"userContext": {}, "newFacts": [], "factsToRemove": []}

    # 构建对话文本
    conversation_text = _format_turns_for_extraction(turns, max_turns=10)

    # 已有事实摘要
    existing_summary = _summarize_existing_facts(existing_facts or [])

    # 纠正/强化提示
    correction_hint = _build_correction_hint(correction_detected, reinforcement_detected)

    prompt = FACT_EXTRACTION_PROMPT.format(
        conversation_text=conversation_text,
        existing_facts_summary=existing_summary,
        correction_hint=correction_hint,
    )

    try:
        from ..utils import get_llm, extract_json_from_llm_response
        llm = get_llm(temperature=0.1, max_tokens=1024)
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke([
            SystemMessage(content="你是科研助手的记忆提取模块。只返回 JSON。"),
            HumanMessage(content=prompt),
        ])
        result = extract_json_from_llm_response(str(response.content))
        return {
            "userContext": result.get("userContext", {}),
            "newFacts": result.get("newFacts", []),
            "factsToRemove": result.get("factsToRemove", []),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.debug("事实提取失败，返回空结果", exc_info=True)
        return {"userContext": {}, "newFacts": [], "factsToRemove": []}


def merge_facts(
    existing: list[dict],
    new_facts: list[dict],
    *,
    confidence_threshold: float = 0.5,
    max_facts: int = 100,
) -> list[dict]:
    """增量合并事实。

    规则:
    1. 按 content（casefold）去重——新事实覆盖旧事实
    2. 置信度 < threshold 的事实丢弃
    3. 按置信度降序排列，超过 max_facts 的裁剪

    Args:
        existing: 已有事实列表
        new_facts: 新提取的事实列表
        confidence_threshold: 最低置信度阈值
        max_facts: 最大保留数量

    Returns:
        合并后的事实列表
    """
    now = datetime.now().isoformat()

    # 建立已有事实的索引 (casefold content -> index)
    existing_index: dict[str, int] = {}
    for i, fact in enumerate(existing):
        key = _fact_content_key(fact.get("content", ""))
        if key:
            existing_index[key] = i

    merged = list(existing)  # 浅拷贝

    for fact in new_facts:
        confidence = _normalize_confidence(fact.get("confidence", 0.5))
        if confidence < confidence_threshold:
            continue

        content = (fact.get("content", "") or "").strip()
        if not content:
            continue

        key = _fact_content_key(content)
        if key is None:
            continue

        new_entry = {
            "content": content,
            "category": fact.get("category", "context"),
            "confidence": confidence,
            "createdAt": now,
            "updatedAt": now,
            "source": "llm_extraction",
        }

        if key in existing_index:
            # 覆盖旧事实
            old_entry = merged[existing_index[key]]
            new_entry["createdAt"] = old_entry.get("createdAt", now)
            # 如果新置信度更高则提升，否则保留旧置信度
            if confidence <= old_entry.get("confidence", 0):
                continue  # 不更新，旧事实置信度更高
            merged[existing_index[key]] = new_entry
        else:
            merged.append(new_entry)

    # 按置信度降序 + 裁剪
    merged.sort(key=lambda f: f.get("confidence", 0), reverse=True)
    if len(merged) > max_facts:
        merged = merged[:max_facts]

    return merged


def remove_facts(existing: list[dict], fact_ids: list[str]) -> list[dict]:
    """删除指定 ID 的事实。

    也支持按内容匹配删除（factsToRemove 可能是 ID 或内容摘要）。
    """
    if not fact_ids:
        return existing

    remove_keys = {_fact_content_key(fid) for fid in fact_ids if isinstance(fid, str)}
    return [
        f for f in existing
        if f.get("id") not in fact_ids
        and _fact_content_key(f.get("id", "")) not in remove_keys
        # 也尝试按 content 匹配
        and _fact_content_key(f.get("content", "")) not in remove_keys
    ]


def format_facts_for_injection(
    facts: list[dict],
    *,
    max_tokens: int = 800,
    top_n: int = 15,
) -> str:
    """将事实格式化为可注入 LLM 上下文的文本。

    按置信度排序，取 top N，控制 token 预算。

    Args:
        facts: 事实列表
        max_tokens: 最大 token 预算（中文约 1.5 字/token）
        top_n: 最多取前 N 条

    Returns:
        格式化的记忆文本，如 "用户偏好 CVD 方法 (置信度:0.95)"
    """
    if not facts:
        return ""

    # 按置信度排序
    sorted_facts = sorted(facts, key=lambda f: f.get("confidence", 0), reverse=True)
    selected = sorted_facts[:top_n]

    lines = []
    char_budget = int(max_tokens * 1.5)  # 中文 token 估算
    char_count = 0

    for fact in selected:
        content = fact.get("content", "")
        confidence = fact.get("confidence", 0)
        category = fact.get("category", "")
        line = f"- [{category}] {content} (置信度:{confidence:.0%})"
        char_count += len(line)
        if char_count > char_budget:
            break
        lines.append(line)

    return "\n".join(lines)


def build_user_profile_text(
    facts: list[dict],
    user_context: dict | None = None,
    unresolved_questions: list[str] | None = None,
) -> str:
    """构建完整的用户画像文本，用于被动注入。

    参考 DeerFlow 的 format_memory_for_injection。
    """
    parts = []

    # 用户上下文
    if user_context:
        ctx_lines = []
        if user_context.get("researchFocus"):
            ctx_lines.append(f"研究方向: {user_context['researchFocus']}")
        if user_context.get("methodPreference"):
            ctx_lines.append(f"方法偏好: {user_context['methodPreference']}")
        if user_context.get("expertiseLevel"):
            ctx_lines.append(f"经验水平: {user_context['expertiseLevel']}")
        if ctx_lines:
            parts.append("## 用户研究画像\n" + "\n".join(ctx_lines))

    # 结构化事实
    facts_text = format_facts_for_injection(facts, max_tokens=800, top_n=15)
    if facts_text:
        parts.append(f"## 已知事实\n{facts_text}")

    # 未解决问题
    if unresolved_questions:
        parts.append("## 上次未解决问题\n" + "\n".join(
            f"- {q}" for q in unresolved_questions[:5]
        ))

    return "\n\n".join(parts)


# ── 内部辅助 ──

def _format_turns_for_extraction(turns: list, max_turns: int = 10) -> str:
    """格式化对话轮次为 LLM 输入。"""
    recent = turns[-max_turns:] if len(turns) > max_turns else turns
    parts = []
    for i, turn in enumerate(recent, 1):
        q = getattr(turn, "question", "")
        a = getattr(turn, "answer", "")
        if q and a:
            parts.append(f"--- 轮次 {i} ---\n问: {q[:200]}\n答: {a[:200]}")
    return "\n\n".join(parts)


def _summarize_existing_facts(facts: list[dict]) -> str:
    """生成已有事实的摘要（避免 LLM 重复提取）。"""
    if not facts:
        return "（暂无已有事实）"

    # 只取前 20 条高置信度的
    sorted_facts = sorted(facts, key=lambda f: f.get("confidence", 0), reverse=True)
    top = sorted_facts[:20]

    lines = []
    for f in top:
        content = f.get("content", "")
        cat = f.get("category", "")
        conf = f.get("confidence", 0)
        lines.append(f"- [{cat}] {content} (置信度:{conf:.0%})")
    return "\n".join(lines)


def _build_correction_hint(
    correction_detected: bool,
    reinforcement_detected: bool,
) -> str:
    """构建纠正/强化提示。"""
    hints = []
    if correction_detected:
        hints.append(
            "**重要**: 检测到用户纠正了AI的回答。请重点关注被纠正的内容，"
            "将正确的信息提取为事实，类别设为 'correction'，置信度 >= 0.90。"
            "同时将对应的过时事实放入 factsToRemove。"
        )
    if reinforcement_detected:
        hints.append(
            "**重要**: 检测到用户肯定了AI的回答。被认可的信息应该提取为事实，"
            "类别设为 'preference' 或 'method_knowledge'，置信度 >= 0.85。"
        )
    return "\n".join(hints)


def _fact_content_key(content: str) -> str | None:
    """生成事实内容的标准化键（用于去重）。"""
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    return stripped.casefold()


def _normalize_confidence(value: Any) -> float:
    """归一化置信度到 [0, 1]。"""
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return max(0.0, min(1.0, float(value)))
    return 0.5
