"""上下文压缩 — 三段式，每轮 LLM 调用前自动触发。

设计（参考 Claude Code s08）:
  L1: tool_result_budget  — 单次工具返回 > 30KB → 存盘留预览
  L2: micro_compact       — 旧 tool_result(>3轮前) → 占位符
  L3: auto_compact        — 总 token 超阈值 → LLM 总结历史

原则: 便宜的在前，贵的在后。L1/L2 零 API 调用。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# 阈值
TOOL_RESULT_BUDGET = 30_000      # 单次 tool_result 超此 → 存盘
MESSAGE_COUNT_LIMIT = 40          # 消息数超此 → 触发 snip
CONTEXT_TOKEN_ESTIMATE = 50_000   # token 估算超此 → LLM 总结
KEEP_RECENT_TOOL_RESULTS = 4     # 保留最近 N 个 tool_result

# 存盘目录
COMPACT_DIR = Path("./data/compact")


def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数: 字符数 / 2.5（中英文混合经验值）。"""
    return sum(len(str(m)) for m in messages) // 2.5


# ═══════════════════════════════════════════════════════════
# L1: tool_result_budget — 大结果存盘
# ═══════════════════════════════════════════════════════════

def _persist_large(tool_use_id: str, content: str) -> str:
    """存盘，返回预览。"""
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    path = COMPACT_DIR / f"tool_{tool_use_id}_{int(time.time())}.txt"
    path.write_text(content, encoding="utf-8")
    preview = content[:2000]
    return (
        f"<persisted-output>\n"
        f"完整内容: {path}\n"
        f"预览(2000字):\n{preview}\n"
        f"</persisted-output>"
    )


def tool_result_budget(messages: list) -> list:
    """L1: 单条 tool_result > 30KB → 存盘留预览。"""
    last = messages[-1] if messages else None
    if not last or not hasattr(last, 'role') or last.role != "user":
        return messages
    if not hasattr(last, 'content') or not isinstance(last.content, list):
        return messages

    for block in last.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        content = str(block.get("content", ""))
        if len(content) > TOOL_RESULT_BUDGET:
            block["content"] = _persist_large(
                block.get("tool_use_id", "unknown"), content
            )
            logger.info("L1压缩: %d → %d 字符", len(content), len(block["content"]))
    return messages


# ═══════════════════════════════════════════════════════════
# L2: micro_compact — 旧 tool_result → 占位符
# ═══════════════════════════════════════════════════════════

def micro_compact(messages: list) -> list:
    """L2: 超过 KEEP 个的旧 tool_result → '[已压缩]'。"""
    result_blocks = []  # (msg_idx, block_idx, block)
    for mi, msg in enumerate(messages):
        if not hasattr(msg, 'content') or not isinstance(msg.content, list):
            continue
        for bi, block in enumerate(msg.content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_blocks.append((mi, bi, block))

    if len(result_blocks) <= KEEP_RECENT_TOOL_RESULTS:
        return messages

    for _, _, block in result_blocks[:-KEEP_RECENT_TOOL_RESULTS]:
        if len(str(block.get("content", ""))) > 120:
            block["content"] = "[已压缩 — 重新执行工具可获取完整结果]"

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
