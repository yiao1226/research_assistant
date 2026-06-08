"""上下文压缩 — 三段式，每轮 LLM 调用前自动触发。

设计（适应 LangChain 消息格式）:
  L1: tool_result_budget  — 单次 ToolMessage > 30KB → 存盘留预览
  L2: micro_compact       — 旧 ToolMessage(>4条前) → 占位符
  L3: auto_compact        — 总 token 超阈值 → LLM 总结历史

原则: 便宜的在前，贵的在后。L1/L2 零 API 调用。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# 阈值
TOOL_RESULT_BUDGET = 30_000      # 单次 ToolMessage 超此 → 存盘
MESSAGE_COUNT_LIMIT = 40          # 消息数超此 → 触发 snip
CONTEXT_TOKEN_ESTIMATE = 50_000   # token 估算超此 → LLM 总结
KEEP_RECENT_TOOL_RESULTS = 4     # 保留最近 N 个 ToolMessage

# 存盘目录
COMPACT_DIR = Path("./data/compact")


def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数: 字符数 / 2.5（中英文混合经验值）。"""
    return sum(len(str(m)) for m in messages) // 2.5


# ═══════════════════════════════════════════════════════════
# L1: tool_result_budget — 大 ToolMessage 存盘
# ═══════════════════════════════════════════════════════════

def _persist_large(tool_call_id: str, content: str) -> str:
    """存盘，返回预览。"""
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = tool_call_id.replace("/", "_").replace("\\", "_")[:40]
    path = COMPACT_DIR / f"tool_{safe_id}_{int(time.time())}.txt"
    path.write_text(content, encoding="utf-8")
    preview = content[:2000]
    return (
        f"<persisted-output>\n"
        f"完整内容: {path}\n"
        f"预览(2000字):\n{preview}\n"
        f"</persisted-output>"
    )


def tool_result_budget(messages: list) -> list:
    """L1: 单条 ToolMessage > 30KB → 存盘留预览。"""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content) if hasattr(msg, 'content') else ""
        if len(content) > TOOL_RESULT_BUDGET:
            tid = getattr(msg, 'tool_call_id', 'unknown')
            messages[i] = ToolMessage(
                content=_persist_large(tid, content),
                tool_call_id=tid,
            )
            logger.info("L1压缩: %d → %d 字符 (tool_call_id=%s)",
                        len(content), len(str(messages[i].content)), tid)
            break  # 只压缩最近一条大结果
    return messages


# ═══════════════════════════════════════════════════════════
# L2: micro_compact — 旧 ToolMessage → 占位符
# ═══════════════════════════════════════════════════════════

def micro_compact(messages: list) -> list:
    """L2: 超过 KEEP 个的旧 ToolMessage → '[已压缩]'。"""
    # 找出所有 ToolMessage 的索引
    tm_indices = [
        i for i, msg in enumerate(messages)
        if isinstance(msg, ToolMessage)
    ]

    if len(tm_indices) <= KEEP_RECENT_TOOL_RESULTS:
        return messages

    # 保留最后 KEEP_RECENT_TOOL_RESULTS 条，其余压缩
    for idx in tm_indices[:-KEEP_RECENT_TOOL_RESULTS]:
        msg = messages[idx]
        content = str(msg.content) if hasattr(msg, 'content') else ""
        if len(content) > 120:
            tid = getattr(msg, 'tool_call_id', '')
            messages[idx] = ToolMessage(
                content="[已压缩 — 重新执行工具可获取完整结果]",
                tool_call_id=tid,
            )

    return messages


# ═══════════════════════════════════════════════════════════
# L3: auto_compact — LLM 总结历史
# ═══════════════════════════════════════════════════════════

_COMPACT_PROMPT = """请总结以下科研助手对话，保留关键信息以便继续工作：

必须保留:
1. 当前任务目标
2. 关键发现和决策
3. 已入库/已搜索的论文（标题）
4. 待完成的工作
5. 用户偏好和约束

对话:
{conversation}

输出压缩摘要（500字以内，中文）。"""


def _summarize(messages: list, llm) -> str:
    """LLM 总结对话。"""
    # 只取最近 30 条消息（太多 LLM 总结质量下降）
    recent = messages[-30:]
    conversation = json.dumps(
        [{"role": getattr(m, "role", "?"), "content": str(getattr(m, "content", ""))[:200]}
         for m in recent],
        ensure_ascii=False,
        default=str,
    )[:8000]

    try:
        response = llm.invoke([
            SystemMessage(content="你是对话压缩专家。简洁准确。"),
            HumanMessage(content=_COMPACT_PROMPT.format(conversation=conversation)),
        ])
        return str(response.content).strip()
    except Exception:
        logger.debug("LLM 总结失败", exc_info=True)
        return "对话过长，已自动压缩（摘要生成失败）。"


def auto_compact(messages: list, llm) -> list:
    """L3: 估算 token > 阈值 → LLM 总结 → 替换为摘要。"""
    if estimate_tokens(messages) < CONTEXT_TOKEN_ESTIMATE:
        return messages

    logger.info("L3压缩触发: %d 条消息, ~%d tokens",
                len(messages), estimate_tokens(messages))

    summary = _summarize(messages, llm)
    # 保留最近 5 条 + 摘要
    return [
        HumanMessage(content=f"[上下文已压缩]\n\n{summary}"),
        *messages[-5:],
    ]


# ═══════════════════════════════════════════════════════════
# 统一入口 — before_llm Hook
# ═══════════════════════════════════════════════════════════

def compaction_hook(messages: list, llm=None) -> None:
    """before_llm Hook: L1 → L2 → L3 三段压缩。

    修改 messages 就地生效，不返回。
    """
    # L1+L2: 零 API 调用
    tool_result_budget(messages)
    micro_compact(messages)

    # L3: 1 次 LLM 调用（仅在超阈值时）
    if llm is not None and estimate_tokens(messages) > CONTEXT_TOKEN_ESTIMATE:
        compacted = auto_compact(messages, llm)
        messages.clear()
        messages.extend(compacted)
