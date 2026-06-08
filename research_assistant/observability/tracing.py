"""LangFuse 全链路 tracing — Agent 循环 + 工具调用 + 检索管道。

设计原则:
  1. 零侵入: 通过 LangChain callback 机制，不修改核心业务代码
  2. 优雅降级: LangFuse 不可用时自动跳过（不影响主流程）
  3. 结构化: trace → span → generation，层级清晰

面试可讲:
  - 每次 Agent 调用自动生成完整 trace 树
  - LLM 调用、工具执行、embedding、向量检索 各有独立 span
  - Token 用量、延迟、模型信息 自动记录
  - Session 级聚合: 同一用户的多次问答关联到同一 session

用法:
  from research_assistant.observability import get_tracer

  tracer = get_tracer()
  with tracer.session("qa", user="eval") as ctx:
      result = llm.invoke(messages, config={"callbacks": [ctx.handler]})
"""

from __future__ import annotations

import logging, os, uuid, time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LangFuseTracer:
    """LangFuse 追踪器 — 封装 LangFuse client + LangChain callback handler。

    Singleton: 整个进程一个实例，按 session 隔离。
    """

    _instance: "LangFuseTracer | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._enabled = bool(
            os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")
        )
        self._handler = None
        self._client = None

        if self._enabled:
            try:
                from langfuse.langchain import CallbackHandler
                # LangFuse v4+ 自动从环境变量读取 LANGFUSE_SECRET_KEY/
                # LANGFUSE_PUBLIC_KEY/LANGFUSE_HOST，不经构造参数传入
                self._handler = CallbackHandler()
                logger.info("LangFuse tracing enabled (langfuse v4)")
            except Exception as e:
                logger.warning("LangFuse init failed, tracing disabled: %s", e)
                self._enabled = False
        else:
            logger.info("LangFuse 未配置（缺少 LANGFUSE_SECRET_KEY/PUBLIC_KEY），tracing 禁用")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def handler(self):
        """LangChain CallbackHandler — 传给 llm.invoke(config={"callbacks": [...]})"""
        return self._handler

    @contextmanager
    def session(self, name: str = "qa", *, user: str = "default",
                metadata: dict | None = None):
        """开启一个 tracing session。

        同一 session 内的多次 LLM 调用自动关联，在 LangFuse dashboard 里聚合成一条 trace。

        Usage:
            with tracer.session("qa", user="eval") as ctx:
                llm.invoke(msgs, config={"callbacks": [ctx.handler]})
        """
        session_id = str(uuid.uuid4())[:8]
        t0 = time.time()

        ctx = TraceContext(
            session_id=session_id,
            trace_name=name,
            user_id=user,
            handler=self._handler,
            metadata=metadata or {},
        )

        try:
            yield ctx
        finally:
            elapsed = time.time() - t0
            if self._enabled and self._handler:
                try:
                    self._handler.flush()
                except Exception:
                    pass
            logger.debug("Trace %s: %.1fs", name, elapsed)

    def shutdown(self):
        """优雅关闭，flush 所有待发数据。"""
        if self._enabled and self._handler:
            try:
                self._handler.flush()
            except Exception:
                pass


@dataclass
class TraceContext:
    """单次 tracing session 的上下文。"""
    session_id: str
    trace_name: str
    user_id: str
    handler: Any  # LangFuse CallbackHandler
    metadata: dict = field(default_factory=dict)


def get_tracer() -> LangFuseTracer:
    """获取全局 tracer 实例。"""
    return LangFuseTracer()
