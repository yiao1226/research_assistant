"""知识库检索工具 — Agent 和 LangGraph 共用。

query_knowledge_base: 本地论文库语义检索（HM25 + Dense + RRF）
query_progress: 用户研究进展记录查询
"""
from __future__ import annotations

from collections import OrderedDict

from langchain_core.tools import tool


def make_kb_tools(storage, username: str) -> list:
    """生成知识库工具（闭包捕获 storage + username）。

    Returns:
        [query_knowledge_base, query_progress]
    """
    from ..rag.retrieval import HybridRetriever

    @tool
    def query_knowledge_base(query: str, limit: int = 5) -> str:
        """搜索本地论文知识库。用来回答科研问题、查找论文中的具体参数/方法/结论。

        Args:
            query: 搜索关键词（建议用学术术语）
            limit: 返回结果数，默认5
        """
        try:
            retriever = HybridRetriever(storage, username)
            papers = retriever.search_papers_expanded(
                query, limit=limit, enable_mqe=True, mqe_expansions=3, enable_hyde=True,
            )
        except Exception:
            try:
                retriever = HybridRetriever(storage, username)
                papers = retriever.search_papers(query, limit=limit)
            except Exception as e:
                return f"（知识库搜索失败: {e}）"

        if not papers:
            return "（未在本地知识库中找到相关论文）"
        return _format_papers(papers[:limit])

    @tool
    def query_progress(query: str, limit: int = 5) -> str:
        """查询用户的研究进展记录。用来回答"我做过什么实验""有什么进展""记录了哪些想法"。

        Args:
            query: 搜索关键词
            limit: 返回结果数，默认5
        """
        try:
            retriever = HybridRetriever(storage, username)
            results = retriever.search_progress(query, limit=limit)
        except Exception as e:
            return f"（进展查询失败: {e}）"

        if not results:
            return "（未找到相关进展记录）"
        items = []
        for p in results[:limit]:
            payload = p.get("payload", {})
            items.append(
                f"- [{payload.get('entry_type', '?')}] {payload.get('title', '?')}: "
                f"{payload.get('content', payload.get('text', ''))[:150]}"
            )
        return "\n".join(items)

    return [query_knowledge_base, query_progress]


def _format_papers(papers: list[dict]) -> str:
    """按 paper_id 分组，格式化论文为 LLM 友好的 Markdown。"""
    groups = OrderedDict()
    for p in papers:
        pid = str(p.get("paper_id") or p.get("id", ""))
        if not pid:
            continue
        text = (
            p.get("text", "") or
            p.get("payload", {}).get("text", "") or
            p.get("payload", {}).get("window_text", "")
        )
        if not text:
            continue
        if pid not in groups:
            groups[pid] = {
                "title": p.get("title", ""),
                "abstract": (p.get("abstract", "") or "")[:150],
                "core": p.get("core_claim", ""),
                "chunks": [],
            }
        if not groups[pid]["title"] and p.get("title"):
            groups[pid]["title"] = p.get("title")
        if not groups[pid]["core"] and p.get("core_claim"):
            groups[pid]["core"] = p.get("core_claim")
        groups[pid]["chunks"].append({
            "heading": p.get("heading_path", ""),
            "text": text[:800],
        })

    items = list(groups.items())[:5]
    lines = []
    for i, (pid, info) in enumerate(items):
        title = info["title"] or "?"
        lines.append(f"\n### [论文{i + 1}] {title}")
        if info["abstract"]:
            lines.append(f"摘要: {info['abstract']}")
        if info["core"]:
            lines.append(f"核心结论: {info['core']}")
        for chunk in info["chunks"][:3]:
            hp = f" ({chunk['heading']})" if chunk["heading"] else ""
            lines.append(f"匹配内容{hp}: {chunk['text']}")
    return "\n".join(lines)
