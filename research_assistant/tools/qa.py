"""QA Agent — 向后兼容 shim，实际实现在 agent/qa.py。"""
from ..agent.qa import QAService, AGENT_SYSTEM_PROMPT

__all__ = ["QAService", "AGENT_SYSTEM_PROMPT"]
