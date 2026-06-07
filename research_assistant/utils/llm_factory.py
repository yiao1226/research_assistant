"""统一 LLM 工厂函数 — 消除各模块中重复的 _build_llm/_make_llm。

整个项目通过此模块获取 ChatOpenAI 实例，配置从 .env 统一读取。
支持实例缓存（按参数组合），避免重复创建。
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


_llm_cache: dict[str, ChatOpenAI] = {}


def _cache_key(model: str, temperature: float, max_tokens: int) -> str:
    return f"{model}_{temperature}_{max_tokens}"


def get_llm(
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model: Optional[str] = None,
    cache: bool = True,
) -> ChatOpenAI:
    """获取统一的 LLM 实例。

    Args:
        temperature: 生成温度 (0.0-1.0)
        max_tokens: 最大输出 token 数
        model: 模型 ID，默认从 LLM_MODEL_ID 环境变量读取
        cache: 是否缓存实例（同参数复用，减少内存占用）

    Returns:
        ChatOpenAI 实例
    """
    if model is None:
        model = os.getenv("LLM_MODEL_ID", "deepseek-chat")

    key = _cache_key(model, temperature, max_tokens)

    if cache and key in _llm_cache:
        return _llm_cache[key]

    instance = ChatOpenAI(
        model=model,
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if cache:
        _llm_cache[key] = instance

    return instance


def clear_llm_cache():
    """清空 LLM 实例缓存（测试用）。"""
    _llm_cache.clear()
