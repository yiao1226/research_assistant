"""统一 JSON 提取工具 — 消除各模块中重复的 LLM 响应 JSON 解析逻辑。

处理 LLM 返回的各种格式:
  - 纯 JSON: {"key": "value"}
  - Markdown 代码块: ```json ... ```
  - 无语言标注代码块: ``` ... ```
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_llm_response(content: str) -> Any:
    """从 LLM 响应中提取 JSON 数据。

    支持格式:
      - 纯 JSON 字符串
      - ```json ... ``` 代码块
      - ``` ... ``` 代码块

    Args:
        content: LLM 响应的原始文本

    Returns:
        解析后的 Python 对象（dict / list / 等）

    Raises:
        json.JSONDecodeError: JSON 解析失败时抛出
    """
    if not content:
        raise json.JSONDecodeError("Empty content", "", 0)

    text = content.strip()

    # 提取 Markdown 代码块中的 JSON
    if "```" in text:
        # 匹配 ```json ... ``` 或 ``` ... ```
        match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # 只有开头的 ```，取其后内容
            parts = text.split("```")
            # 跳过空的第一部分，取第二部分
            for part in parts[1:]:
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part:
                    text = part
                    break

    return json.loads(text)


def try_extract_json(content: str, default: Any = None) -> Any:
    """尝试从 LLM 响应提取 JSON，失败时返回默认值。

    Args:
        content: LLM 响应的原始文本
        default: 解析失败时的默认返回值

    Returns:
        解析后的对象或默认值
    """
    try:
        return extract_json_from_llm_response(content)
    except (json.JSONDecodeError, AttributeError):
        return default
