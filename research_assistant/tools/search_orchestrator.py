"""搜索编排器 — Query Understanding + 多源搜索 + LLM排序 + Top-N摘要。

流程:
  1. LLM Query Understanding（关键词扩展 + 搜索策略）
  2. 多源并行搜索（ArXiv + Semantic Scholar）
  3. 合并去重 → LLM 重排序（替代固定权重 40/25/20/15）
  4. LLM 生成 Top-N 摘要 + 排名理由

sort_by 可配置: relevance / recency / citations
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from .paper_search import search_arxiv, search_semantic_scholar, search_web_of_science
from ..core.storage import PerUserStorage
from ..utils import get_llm
from ..schemas import QueryExpansionResult, RankingResult

QUERY_UNDERSTANDING_PROMPT = """你是学术搜索策略专家。将查询扩展为更全面的搜索策略。

用户研究背景:
- 研究主题: {topics}
- 最近进展: {recent_progress}
- 已有论文方向: {paper_directions}
- 知识缺口: {gaps}

原始查询: "{raw_query}"

输出 JSON 搜索策略:
{{
  "expanded_queries": [
    {{"query": "扩展搜索词(英文, 支持布尔)", "rationale": "扩展理由", "priority": "high/medium/low"}}
  ],
  "search_focus": "搜索重点",
  "exclude_directions": "排除方向"
}}
优先扩展知识缺口方向。只返回 JSON。"""

RANKING_PROMPT = """你是学术论文分析专家。对搜索结果排序并分析。

查询: "{query}"
排序偏好: {sort_by}

候选论文:
{papers_text}

对每篇论文提取:
- core_contribution: 核心贡献（1句话）
- innovation: 创新点
- relevance_reason: 与查询相关性
- composite_score: 0-100评分

以 {sort_by} 为主要排序依据，返回 JSON:
{{"ranked": [{{"index": 序号(1-based), "core_contribution": "...", "innovation": "...", "relevance_reason": "...", "ranking_reason": "..."}}]}}
只返回 JSON。"""


@dataclass
class SearchContext:
    """用户搜索上下文（研究画像）。"""
    topics: list[str] = field(default_factory=list)
    recent_progress: list[str] = field(default_factory=list)
    paper_directions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


class SearchOrchestrator:
    """搜索编排器 — LLM排序，sort_by可配置。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username

    # === Step 1: 构建上下文 ===

    def build_search_context(self) -> SearchContext:
        """从用户数据构建搜索画像。"""
        ctx = SearchContext()

        progress = self.storage.get_all_progress(limit=10)
        for p in progress:
            title = p.get("title", "")
            insights = p.get("insights", "")
            if title:
                ctx.recent_progress.append(f"{title}: {insights}" if insights else title)

        papers = self.storage.get_all_papers()
        topic_set = set()
        for p in papers[:50]:
            ann = p.get("annotation")
            if isinstance(ann, str):
                try:
                    ann = json.loads(ann)
                except json.JSONDecodeError:
                    ann = {}
            if isinstance(ann, dict):
                for kw in ann.get("keywords_material", [])[:2]:
                    if kw:
                        topic_set.add(kw)
                for kw in ann.get("keywords_phenomenon", [])[:2]:
                    if kw:
                        topic_set.add(kw)
        ctx.paper_directions = list(topic_set)[:15]

        topics: dict[str, int] = {}
        for p in progress[:10]:
            t = p.get("topic", "")
            if t:
                topics[t] = topics.get(t, 0) + 1
        ctx.topics = sorted(topics, key=topics.get, reverse=True)[:3]

        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            ctx.gaps = em.get_unresolved_questions(limit=5)
        except Exception:
            pass

        return ctx

    # === Step 2: Query Understanding ===

    def expand_query(self, raw_query: str, ctx: SearchContext) -> list[dict]:
        """LLM 驱动关键词扩展（Pydantic 结构化输出，自动校验）。"""
        llm = get_llm(temperature=0.2)
        prompt = QUERY_UNDERSTANDING_PROMPT.format(
            topics="、".join(ctx.topics) if ctx.topics else "未指定",
            recent_progress="; ".join(ctx.recent_progress[-5:]) if ctx.recent_progress else "无",
            paper_directions="、".join(ctx.paper_directions[:10]) if ctx.paper_directions else "无",
            gaps="; ".join(ctx.gaps) if ctx.gaps else "无",
            raw_query=raw_query,
        )
        try:
            structured_llm = llm.with_structured_output(QueryExpansionResult)
            result = structured_llm.invoke([
                SystemMessage(content="你是学术搜索策略专家。"),
                HumanMessage(content=prompt),
            ])
            return [{"query": q.query, "rationale": q.rationale, "priority": q.priority}
                    for q in result.expanded_queries]
        except Exception:
            return [{"query": raw_query, "rationale": "原始查询", "priority": "high"}]

    # === Step 3: 多源搜索 ===

    def search_all_sources(self, queries: list[dict], sources: list[str]) -> list[dict]:
        """多源多关键词并行搜索。"""
        all_papers: list[dict] = []
        seen: set[str] = set()

        sorted_queries = sorted(queries, key=lambda q: (
            0 if q.get("priority") == "high" else 1 if q.get("priority") == "medium" else 2
        ))

        def _search_one(src_key: str, query_str: str, max_r: int) -> list[dict]:
            try:
                func = {"arxiv": search_arxiv, "s2": search_semantic_scholar, "wos": search_web_of_science}.get(src_key)
                if not func:
                    return []
                result = func.invoke({"query": query_str, "max_results": max_r})
                data = json.loads(result) if isinstance(result, str) else result
                return data.get("papers", []) if data.get("status") == "ok" else []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=max(1, len(sources))) as executor:
            for q in sorted_queries[:5]:
                query_str = q["query"]
                max_r = 10 if q.get("priority") == "high" else 5
                futures = {executor.submit(_search_one, src, query_str, max_r): src for src in sources}
                for future in as_completed(futures):
                    for paper in future.result():
                        key = paper.get("arxiv_id") or paper.get("doi") or paper.get("title", "")
                        if key and key in seen:
                            continue
                        if key:
                            seen.add(key)
                        paper["search_priority"] = q.get("priority", "medium")
                        all_papers.append(paper)
        return all_papers

    # === Step 4: LLM 重排序（替代固定权重） ===

    def rank_papers(self, papers: list[dict], query: str, sort_by: str = "relevance") -> list[dict]:
        """LLM 对结果排序 + 提取核心信息。

        替代旧版固定权重（语义40%+关键词25%+热门20%+时效15%）。
        LLM 综合判断相关性/时效/引用，更灵活。
        """
        if len(papers) <= 5:
            return self._score_by_basic(papers, sort_by)

        llm = get_llm(temperature=0.2)
        papers_text = "\n\n".join(
            f"[{i+1}] {p.get('title', '?')}\n"
            f"    作者: {', '.join(p.get('authors', [])[:3])}\n"
            f"    年份: {p.get('year', '?')} 引用: {p.get('citation_count', 0) or 0}\n"
            f"    摘要: {(p.get('abstract', '') or '')[:300]}"
            for i, p in enumerate(papers[:15])
        )
        prompt = RANKING_PROMPT.format(query=query, sort_by=sort_by, papers_text=papers_text)

        try:
            structured_llm = llm.with_structured_output(RankingResult)
            result = structured_llm.invoke([
                SystemMessage(content="你是学术论文检索排序专家。"),
                HumanMessage(content=prompt),
            ])
            for item in result.ranked:
                idx = item.index - 1
                if 0 <= idx < len(papers):
                    papers[idx]["core_contribution"] = item.core_contribution
                    papers[idx]["innovation"] = item.innovation
                    papers[idx]["relevance_reason"] = item.relevance_reason
                    papers[idx]["ranking_reason"] = item.ranking_reason
                    papers[idx]["composite_score"] = 100 - idx * 5  # LLM排序位置→分数
            papers.sort(key=lambda p: p.get("composite_score", 0), reverse=True)
        except Exception:
            papers = self._score_by_basic(papers, sort_by)

        return papers

    def _score_by_basic(self, papers: list[dict], sort_by: str) -> list[dict]:
        """轻量排序: 小结果集或 LLM 失败时兜底。"""
        for p in papers:
            citations = int(p.get("citation_count", 0) or 0)
            year = int(p.get("year", 0) or 0)
            score = citations * 0.5 + year * 0.5 if sort_by == "citations" else citations * 0.3 + year * 0.3
            p["composite_score"] = round(score, 1)
        return sorted(papers, key=lambda p: p.get("composite_score", 0), reverse=True)

    # === 完整流程 ===

    def search(self, raw_query: str, sources: list[str] | None = None,
               sort_by: str = "relevance") -> dict:
        """执行完整搜索流程。

        Args:
            raw_query: 用户原始查询
            sources: 搜索源（默认 arxiv + s2）
            sort_by: "relevance" / "recency" / "citations"

        Returns:
            {"papers": [...], "total_found": int, "search_focus": str,
             "expanded_queries": [...], "duration_sec": float}
        """
        if sources is None:
            sources = ["arxiv", "s2"]
            if os.getenv("WOS_API_KEY"):
                sources.append("wos")

        t_start = time.time()

        # Step 1: 用户画像
        ctx = self.build_search_context()

        # Step 2: Query Understanding
        expanded = self.expand_query(raw_query, ctx)
        if not expanded:
            expanded = [{"query": raw_query, "rationale": "原始查询", "priority": "high"}]

        search_focus = next(
            (e.get("rationale", raw_query) for e in expanded if e.get("priority") == "high"),
            raw_query,
        )

        # Step 3: 多源搜索
        all_papers = self.search_all_sources(expanded, sources)

        # Step 4: LLM 排序 + 摘要注入
        ranked = self.rank_papers(all_papers, raw_query, sort_by)

        duration = time.time() - t_start

        return {
            "papers": ranked[:5],
            "total_found": len(all_papers),
            "search_focus": search_focus,
            "expanded_queries": expanded,
            "duration_sec": duration,
        }
