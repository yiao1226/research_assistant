"""Chat endpoint schemas — SSE 事件类型定义。

SSE 事件流设计（面试要点）:
  不是简单的 text/event-stream，而是类型化的 JSON 事件。
  前端可以根据 event 类型做差异化渲染:
    - thinking → spinner
    - tool_call → 显示工具调用卡片
    - token → 打字机效果
    - citation → 论文引用列表
    - done → 关闭连接/显示统计
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class ChatRequest(BaseModel):
    """POST /chat/stream 请求体"""
    message: str = Field(..., min_length=1, max_length=5000, description="用户问题")
    username: str = Field(default="eval", description="用户名")


class SSEEvent(BaseModel):
    """SSE 事件 — 统一的事件模型"""
    event: Literal["thinking", "tool_call", "tool_result", "token",
                   "citation", "error", "done"]
    data: dict = Field(default_factory=dict)


class ThinkingData(BaseModel):
    """Agent 开始思考"""
    message: str


class ToolCallData(BaseModel):
    """工具被调用"""
    tool: str
    args: dict


class ToolResultData(BaseModel):
    """工具执行完成"""
    tool: str
    summary: str


class TokenData(BaseModel):
    """逐 token 输出"""
    delta: str


class CitationData(BaseModel):
    """引用论文"""
    papers: list[dict]


class DoneData(BaseModel):
    """回答完成"""
    tool_calls: int = 0
    cited_papers: int = 0


class ErrorData(BaseModel):
    """错误"""
    code: str
    message: str
