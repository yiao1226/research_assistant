"""外部论文搜索工具 — ArXiv + Semantic Scholar。

search_papers_online: 搜索 + 展示 + 交互选择入库
   被 Agent 使用（交互模式），LangGraph research 节点使用（非交互模式）。
"""
from __future__ import annotations

from langchain_core.tools import tool

# 跨工具共享: 最近一次搜索结果缓存（供 ingest 工具使用）
_last_search_papers: list[dict] = []


def get_last_search_results() -> list[dict]:
    return _last_search_papers


def make_search_tool(storage, username: str):
    """生成外部搜索 + 入库工具组。"""

    @tool
    def search_papers_online(query: str) -> str:
        """从 ArXiv + Semantic Scholar 搜索新论文。
        搜完后向用户展示结果，询问是否入库——不要直接写综述。

        Args:
            query: 搜索关键词（英文效果更好）
        """
        global _last_search_papers
        import sys

        try:
            print(f"     ⏳ 搜索 ArXiv + Semantic Scholar...", flush=True)
            from ..tools.search_orchestrator import SearchOrchestrator
            orch = SearchOrchestrator(storage, username)
            result = orch.search(query, sources=["arxiv", "s2"])
            papers = result.get("papers", [])
            _last_search_papers = papers

            if not papers:
                return "（未找到相关论文，换个关键词试试）"

            # 展示结果
            total = result.get('total_found', len(papers))
            duration = result.get('duration_sec', 0)
            print(f"\n  {'─'*55}")
            print(f"  🔍 找到 {total} 篇, 精选 Top-{len(papers)} | 耗时 {duration:.1f}s")
            print(f"  {'─'*55}")
            for i, p in enumerate(papers[:5], 1):
                score = p.get('composite_score', 0)
                cites = p.get('citation_count', 0) or 0
                year = p.get('year', '')
                title = p.get('title', '?')
                core = p.get('core_contribution', '')[:120]
                print(f"\n  [{i}] {title}")
                print(f"      {score:.0f}/100 | 引用: {cites} | {year}")
                print(f"      {core}")

            # 交互选择
            print(f"\n  [I] 入库 (如 I1,3)  [D] 下载 (如 D1)  [S] 跳过  [Q] 追问")
            choice = input("  > ").strip()

            if choice.lower() == 's' or not choice:
                return _format_result(papers, "用户选择跳过")

            if choice.lower() == 'q':
                follow_up = input("  想问什么？> ").strip()
                return _format_result(papers, f"用户追问: {follow_up}")

            # 解析 I/D 选项
            ingest_nums, download_nums = _parse_choice(choice, len(papers))
            messages = []

            if ingest_nums:
                try:
                    from ..rag.ingestion import IngestionPipeline
                    pipeline = IngestionPipeline(storage, username)
                except Exception:
                    pipeline = None

                for idx in ingest_nums:
                    p = papers[idx - 1]
                    if pipeline:
                        try:
                            pid = pipeline.ingest(p)
                            print(f"     ✅ 已入库 [{idx}]: {p.get('title', '?')[:50]} (ID={pid})")
                            messages.append(f"已入库 [{idx}] {p.get('title', '')[:80]}")
                        except Exception as e:
                            print(f"     ❌ 入库失败 [{idx}]: {e}")
                            messages.append(f"入库失败 [{idx}]: {e}")
                    else:
                        messages.append(f"[{idx}] {p.get('title', '')[:80]} (需入库管线)")

            if download_nums:
                for idx in download_nums:
                    p = papers[idx - 1]
                    arxiv_id = p.get("arxiv_id", "")
                    if arxiv_id:
                        try:
                            from ..tools.paper_search import download_paper
                            r = download_paper.invoke({"arxiv_id": arxiv_id})
                            print(f"     📥 [{idx}] {r}")
                        except Exception:
                            messages.append(f"下载失败 [{idx}]")
                    else:
                        print(f"     ⚠ [{idx}] 无 ArXiv ID, 无法下载")

            return _format_result(papers, "; ".join(messages) if messages else "用户查看了搜索结果")

        except Exception as e:
            return f"（外部搜索暂时不可用: {e}）"

    @tool
    def ingest_papers(choices: str) -> str:
        """入库最近一次搜索到的论文。只在 search_papers_online 之后使用。

        Args:
            choices: 要入库的编号，如 "1,3,5" 或 "all"
        """
        papers = get_last_search_results()
        if not papers:
            return "（没有搜索结果可入库，请先搜索）"

        if choices.lower() == "all":
            indices = list(range(1, len(papers) + 1))
        else:
            indices = _parse_numbers(choices, len(papers))

        if not indices:
            return f"（无效选择: {choices}，可用: 1-{len(papers)}）"

        try:
            from ..rag.ingestion import IngestionPipeline
            pipeline = IngestionPipeline(storage, username)
        except Exception:
            return "（入库管线不可用）"

        ingested = []
        failed = []
        for idx in indices:
            p = papers[idx - 1]
            try:
                pid = pipeline.ingest(p)
                ingested.append(f"[{idx}] {p.get('title', '')[:60]} (ID={pid})")
            except Exception as e:
                failed.append(f"[{idx}] {e}")

        parts = []
        if ingested:
            parts.append(f"已入库 {len(ingested)} 篇:\n" + "\n".join(ingested))
        if failed:
            parts.append(f"失败 {len(failed)} 篇:\n" + "\n".join(failed))
        return "\n\n".join(parts) if parts else "（无操作）"

    return [search_papers_online, ingest_papers]


# ── 辅助 ──

def _parse_choice(choice: str, max_n: int) -> tuple[list[int], list[int]]:
    """解析 "I1,3 D2" 格式的选择。"""
    ingest, download = [], []
    for part in choice.upper().replace(",", " ").split():
        if part.startswith("I"):
            nums = _parse_numbers(part[1:], max_n)
            ingest.extend(nums)
        elif part.startswith("D"):
            nums = _parse_numbers(part[1:], max_n)
            download.extend(nums)
    return ingest, download


def _parse_numbers(s: str, max_n: int) -> list[int]:
    """解析 "1,3,5" → [1, 3, 5]。"""
    nums = []
    for token in s.replace(",", " ").split():
        try:
            n = int(token)
            if 1 <= n <= max_n:
                nums.append(n)
        except ValueError:
            pass
    return nums


def _format_result(papers: list[dict], action: str) -> str:
    """格式化搜索结果为 Agent 友好文本。"""
    lines = [f"搜索结果: 共 {len(papers)} 篇 | {action}"]
    for i, p in enumerate(papers[:5], 1):
        lines.append(
            f"[{i}] {p.get('title', '?')}\n"
            f"    评分: {p.get('composite_score', 0):.0f}/100 | "
            f"引用: {p.get('citation_count', 0) or 0} | {p.get('year', '')}\n"
            f"    核心: {p.get('core_contribution', '')[:200]}"
        )
    return "\n".join(lines)
