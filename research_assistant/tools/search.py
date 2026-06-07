"""外部论文搜索工具 — ArXiv + Semantic Scholar。

search_papers_online: 从外部学术平台搜索新论文
   被 Agent 和 LangGraph research 节点共用。
"""
from __future__ import annotations

from langchain_core.tools import tool


def make_search_tool(storage, username: str):
    """生成外部搜索工具（闭包捕获 storage + username）。"""

    @tool
    def search_papers_online(query: str) -> str:
        """从 ArXiv + Semantic Scholar 搜索新论文。
        用来回答"找一下XX的最新论文""搜索XX领域""有没有关于XX的研究"。

        Args:
            query: 搜索关键词（英文效果更好）
        """
        try:
            from ..tools.search_orchestrator import SearchOrchestrator
            orch = SearchOrchestrator(storage, username)
            result = orch.search(query, sources=["arxiv", "s2"])
            papers = result.get("papers", [])
            if not papers:
                return (
                    f"外部搜索 '{query}': 未找到结果。"
                    f"搜索策略: {result.get('search_focus', '')}"
                )
            lines = [
                f"搜索 '{query}': 找到 {result.get('total_found', 0)} 篇, "
                f"精选 Top-{len(papers)}, 耗时 {result.get('duration_sec', 0):.1f}s\n",
            ]
            for i, p in enumerate(papers[:5], 1):
                lines.append(
                    f"[{i}] {p.get('title', '?')}\n"
                    f"    评分: {p.get('composite_score', 0):.0f}/100 | "
                    f"引用: {p.get('citation_count', 0) or 0} | "
                    f"年份: {p.get('year', '')}\n"
                    f"    核心: {p.get('core_contribution', '')[:200]}\n"
                    f"    摘要: {(p.get('abstract', '') or '')[:300]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"（外部搜索暂时不可用: {e}）"

    return search_papers_online
