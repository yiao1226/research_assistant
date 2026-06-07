"""LangGraph 工作流 — 向后兼容 shim，实现在 agent/graph.py。"""
from .agent.graph import build_research_graph

__all__ = ["build_research_graph"]
