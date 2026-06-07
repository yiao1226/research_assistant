"""消息过滤与信号检测 — 参考 DeerFlow 的 message_processing 模块。

提供:
  - detect_correction: 检测用户是否在纠正 AI（"不对""应该是"）
  - detect_reinforcement: 检测用户是否在肯定 AI（"很好""对"）
  - filter_user_messages: 从对话轮次中提取纯用户消息文本
"""

from __future__ import annotations

import re

# ── 纠正信号 ──
# 用户明确指出 AI 的回答有误
_CORRECTION_PATTERNS = [
    r"不对[，。！]",
    r"不是[，。！]",
    r"错了",
    r"应该是",
    r"改一下",
    r"纠正",
    r"不对的",
    r"不是这样",
    r"你搞错了",
    r"说错了",
    r"不是.{0,3}是",
    r"实际上",
    r"准确地说",
    r"更正",
]

# ── 强化（肯定）信号 ──
# 用户明确认可 AI 的回答
_REINFORCEMENT_PATTERNS = [
    r"很好[，。！]",
    r"非常好",
    r"很棒",
    r"很对",
    r"正确",
    r"谢谢",
    r"感谢",
    r"就是这样",
    r"没错",
    r"完全正确",
    r"太好了",
    r"好[，。！]这样就",
    r"可以[，。！]这样就",
    r"对的",
    r"正解",
    r"厉害了",
]


def detect_correction(text: str) -> bool:
    """检测用户消息是否包含纠正信号。

    Args:
        text: 用户原始消息文本

    Returns:
        True 如果检测到用户正在纠正 AI 的回答
    """
    if not text or not isinstance(text, str):
        return False
    for pattern in _CORRECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def detect_reinforcement(text: str) -> bool:
    """检测用户消息是否包含强化/认可信号。

    Args:
        text: 用户原始消息文本

    Returns:
        True 如果检测到用户正在肯定 AI 的回答
    """
    if not text or not isinstance(text, str):
        return False
    for pattern in _REINFORCEMENT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def filter_user_messages(turns: list, last_n: int = 5) -> str:
    """从最近 N 轮对话中提取纯用户消息文本。

    用于 LLM 记忆提取时只关注"用户说了什么"，
    过滤掉工具调用、检索中间结果等噪音。

    Args:
        turns: QATurn 对象列表
        last_n: 最多取最近 N 轮

    Returns:
        格式化的用户消息文本
    """
    if not turns:
        return ""

    recent = turns[-last_n:] if len(turns) > last_n else turns
    lines = []
    for i, turn in enumerate(recent, 1):
        q = getattr(turn, "question", "")
        if q:
            lines.append(f"[轮次{i}] 用户: {q}")
    return "\n".join(lines)


def classify_turn(question: str) -> dict:
    """对单轮对话进行分类，检测信号。

    Returns:
        {
            "correction_detected": bool,
            "reinforcement_detected": bool,
        }
    """
    return {
        "correction_detected": detect_correction(question),
        "reinforcement_detected": detect_reinforcement(question),
    }
