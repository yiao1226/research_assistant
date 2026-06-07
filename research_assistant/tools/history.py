"""情景记忆检索工具 — 历史会话回忆。

recall_history: 搜索历史会话讨论（语义 × 时间衰减）
   被 Agent 和 LangGraph understand 节点共用。
"""
from __future__ import annotations

from langchain_core.tools import tool


def make_history_tool(storage, username: str):
    """生成历史回忆工具（闭包捕获 storage + username）。"""

    @tool
    def recall_history(query: str, limit: int = 3) -> str:
        """搜索历史会话和讨论记录。
        用来回答"上次讨论了XX""之前分析过YY""还记得吗"。

        Args:
            query: 搜索关键词（提取你想回顾的主题词）
            limit: 返回结果数，默认3
        """
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(storage, username)
            results = em.recall_context_hybrid(query, limit=limit, decay_days=30.0)
            if not results:
                return "（未找到相关历史讨论记录）"
            items = []
            for r in results[:limit]:
                text = (r.get("text", "") or r.get("payload", {}).get("text", ""))[:150]
                date_str = r.get("payload", {}).get("session_date", "")
                if text:
                    items.append(f"- [{date_str}] {text}")
            return "\n".join(items) if items else "（未找到相关历史讨论）"
        except Exception as e:
            return f"（历史回忆失败: {e}）"

    return recall_history
