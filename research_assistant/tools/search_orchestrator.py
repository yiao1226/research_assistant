"""搜索编排器 — Query Understanding + 多源搜索 + 多因素排序 + Top-5 摘要。

流程:
  1. 加载用户研究画像（进展 + 论文 + 计划）
  2. LLM Query Understanding（关键词扩展 + 搜索策略）
  3. 多源并行搜索
  4. 合并去重 + 多因素排序
  5. LLM 生成 Top-5 摘要 + 排名理由
  6. 返回结果，等待用户选择入库
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .paper_search import search_arxiv, search_semantic_scholar, search_web_of_science
from ..rag.embedding import EmbeddingService
from ..rag.vector_store import VectorStore
from ..core.storage import PerUserStorage
from ..utils import get_llm, extract_json_from_llm_response

QUERY_UNDERSTANDING_PROMPT = """你是学术搜索策略专家。基于用户的研究背景，将查询扩展为更全面的搜索策略。

用户研究画像:
- 研究主题: {topics}
- 最近进展: {recent_progress}
- 已有论文方向: {paper_directions}
- 知识缺口: {gaps}

用户原始查询: "{raw_query}"

请以 JSON 格式输出搜索策略:
{{
  "expanded_queries": [
    {{
      "query": "扩展后的搜索词（英文，支持布尔运算）",
      "rationale": "为什么扩展为这个词",
      "priority": "high/medium/low"
    }}
  ],
  "search_focus": "本次搜索应该重点关注什么",
  "exclude_directions": "应排除的方向（如有）"
}}

优先扩展与用户知识缺口相关的方向。只返回 JSON。"""

RANKING_PROMPT = """你是学术论文分析专家。基于论文的标题和完整摘要，提取每篇论文的精髓。

查询: "{query}"
搜索策略: {search_focus}

候选论文:
{papers_text}

对每篇论文，基于标题和摘要提取:
- core_contribution: 核心贡献（解决了什么问题，1句话）
- innovation: 创新点（与现有方法有何不同）
- method_brief: 方法简述（用了什么技术/工艺）
- relevance_reason: 与查询的相关性说明
- ranking_reason: 为什么排在这个位置

返回 JSON:
{{"ranked": [
  {{
    "index": 候选列表中的序号(1-based),
    "core_contribution": "核心贡献",
    "innovation": "创新点",
    "method_brief": "方法简述",
    "relevance_reason": "相关性",
    "ranking_reason": "排名理由"
  }}
]}}

只返回 JSON。"""


@dataclass
class SearchContext:
    """用户搜索上下文（研究画像）。"""
    topics: list[str] = field(default_factory=list)
    recent_progress: list[str] = field(default_factory=list)
    paper_directions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    plan_phases: list[str] = field(default_factory=list)


class SearchOrchestrator:
    """搜索编排器 — 个性化搜索 + 多因素排序。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.vector_store = VectorStore()
        self.embedder = EmbeddingService()

    # === Step 1: 构建用户研究画像 ===

    def build_search_context(self, query: str) -> SearchContext:
        """从用户数据构建搜索上下文。"""
        ctx = SearchContext()

        # 最近进展
        progress = self.storage.get_all_progress(limit=10)
        for p in progress:
            title = p.get("title", "")
            insights = p.get("insights", "")
            if title:
                ctx.recent_progress.append(f"{title}: {insights}" if insights else title)

        # 已有论文方向（从标注中提取）
        papers = self.storage.get_all_papers()
        topic_set = set()
        for p in papers[:50]:
            annotation = p.get("annotation")
            if isinstance(annotation, str):
                try:
                    annotation = json.loads(annotation)
                except json.JSONDecodeError:
                    annotation = {}
            if isinstance(annotation, dict):
                for kw in annotation.get("keywords_material", [])[:2]:
                    if kw:
                        topic_set.add(kw)
                for kw in annotation.get("keywords_phenomenon", [])[:2]:
                    if kw:
                        topic_set.add(kw)
        ctx.paper_directions = list(topic_set)[:15]

        # 活跃研究计划
        # 根据进展推断当前主题
        topics = {}
        for p in progress[:10]:
            t = p.get("topic", "")
            if t:
                topics[t] = topics.get(t, 0) + 1
        ctx.topics = sorted(topics, key=topics.get, reverse=True)[:3]

        # 从未解决的历史会话问题中获取知识缺口
        try:
            from ..memory.episodic import EpisodicMemory
            em = EpisodicMemory(self.storage, self.username)
            ctx.gaps = em.get_unresolved_questions(limit=5)
        except Exception:
            pass

        return ctx

    # === Step 2: Query Understanding ===

    def expand_query(self, raw_query: str, ctx: SearchContext) -> list[dict]:
        """LLM 驱动的关键词扩展。"""
        llm = get_llm(temperature=0.2)

        prompt = QUERY_UNDERSTANDING_PROMPT.format(
            topics="、".join(ctx.topics) if ctx.topics else "未指定",
            recent_progress="; ".join(ctx.recent_progress[-5:]) if ctx.recent_progress else "无",
            paper_directions="、".join(ctx.paper_directions[:10]) if ctx.paper_directions else "无",
            gaps="; ".join(ctx.gaps) if ctx.gaps else "无",
            raw_query=raw_query,
        )

        try:
            response = llm.invoke([
                SystemMessage(content="你是学术搜索策略专家。只返回 JSON。"),
                HumanMessage(content=prompt),
            ])
            result = extract_json_from_llm_response(str(response.content))
            return result.get("expanded_queries", [])
        except (json.JSONDecodeError, AttributeError):
            return [{"query": raw_query, "rationale": "原始查询", "priority": "high"}]

    # === Step 3: 多源搜索 ===

    def search_all_sources(self, queries: list[dict], sources: list[str]) -> list[dict]:
        """多源多关键词并行搜索。

        Args:
            queries: [{"query": "...", "priority": "high"}, ...]
            sources: ["arxiv", "s2", "wos"]

        Returns:
            去重后的论文列表
        """
        all_papers = []
        seen = set()

        # 优先级高的查询先搜
        sorted_queries = sorted(queries, key=lambda q: (
            0 if q.get("priority") == "high" else 1 if q.get("priority") == "medium" else 2
        ))

        def _search_one_source(src_key, query_str, max_r):
            """单个搜索源调用（用于线程池）。"""
            try:
                if src_key == "arxiv":
                    result = search_arxiv.invoke({"query": query_str, "max_results": max_r})
                elif src_key == "s2":
                    result = search_semantic_scholar.invoke({"query": query_str, "max_results": max_r})
                elif src_key == "wos":
                    result = search_web_of_science.invoke({"query": query_str, "max_results": max_r})
                else:
                    return []
                data = json.loads(result) if isinstance(result, str) else result
                if data.get("status") != "ok":
                    return []
                return data.get("papers", [])
            except Exception:
                return []

        for q in sorted_queries[:5]:  # 最多 5 个扩展查询
            query_str = q["query"]
            max_r = 10 if q.get("priority") == "high" else 5

            # 并行搜索多个源
            with ThreadPoolExecutor(max_workers=len(sources)) as executor:
                futures = {
                    executor.submit(_search_one_source, src, query_str, max_r): src
                    for src in sources
                }
                for future in as_completed(futures):
                    for paper in future.result():
                        # 去重
                        key = paper.get("arxiv_id") or paper.get("doi") or paper.get("title", "")
                        if key and key in seen:
                            continue
                        if key:
                            seen.add(key)
                        # 标记搜索优先级
                        paper["search_priority"] = q.get("priority", "medium")
                        all_papers.append(paper)

        return all_papers

    # === Step 4: 多因素排序 ===

    def rank_papers(self, papers: list[dict], query: str, ctx: SearchContext) -> list[dict]:
        """多因素排序: 语义相似度 + 关键词 + 热门度 + 时效。

        Returns:
            按 composite_score 降序排列的论文列表
        """
        if not papers:
            return []

        # 构建用户上下文文本（用于语义匹配）
        user_context = " ".join(
            ctx.topics +
            [q for q in ctx.paper_directions[:10]] +
            [q for q in ctx.gaps[:5]]
        )

        # 查询向量 + 批量编码所有论文（避免 N 次独立前向传播）
        query_vec = self.embedder.encode_query(query)

        paper_texts = [
            f"{p.get('title', '')} {p.get('abstract', '')[:300]}"[:1000]
            for p in papers
        ]
        paper_vecs = self.embedder.encode(paper_texts) if paper_texts else []

        scored = []
        current_year = datetime.now().year

        for i, paper in enumerate(papers):
            score = 0.0
            reasons = []

            # 1. 语义匹配度 (40%) — 论文 vs 用户上下文
            if paper_texts[i].strip() and i < len(paper_vecs):
                paper_vec = paper_vecs[i]
                # cosine similarity (vectors are normalized)
                semantic_score = sum(a * b for a, b in zip(query_vec, paper_vec))
                score += 0.40 * min(1.0, max(0.0, semantic_score))
                reasons.append(f"语义匹配: {semantic_score:.2f}")

            # 2. 关键词匹配 (25%) — BM25-like 词频
            kw_score = self._keyword_match(query, paper)
            score += 0.25 * kw_score
            reasons.append(f"关键词: {kw_score:.2f}")

            # 3. 热门程度 (20%) — 引用数 / 年龄
            citations = int(paper.get("citation_count", 0) or 0)
            year = int(paper.get("year", 0) or 0)
            if year > 0 and year <= current_year:
                age = max(1, current_year - year)
                pop_score = min(1.0, citations / (age * 10 + 1))
            else:
                pop_score = 0.1
            score += 0.20 * pop_score
            reasons.append(f"热门: {pop_score:.2f} (引用{citations})")

            # 4. 时效性 (15%) — 越新越好
            if year >= current_year:
                recency = 1.0
            elif year >= current_year - 2:
                recency = 0.8
            elif year >= current_year - 5:
                recency = 0.5
            else:
                recency = 0.2
            score += 0.15 * recency
            reasons.append(f"时效: {recency:.2f} ({year})")

            paper["composite_score"] = round(score * 100, 1)
            paper["score_breakdown"] = "; ".join(reasons)

            scored.append(paper)

        scored.sort(key=lambda p: p["composite_score"], reverse=True)
        return scored

    def _keyword_match(self, query: str, paper: dict) -> float:
        """计算关键词匹配度。"""
        keywords = query.lower().replace("and", " ").replace("or", " ").split()
        if not keywords:
            return 0.0

        text = (
            f"{paper.get('title', '')} "
            f"{paper.get('abstract', '')} "
            f"{paper.get('venue', '')}"
        ).lower()

        hits = sum(1 for kw in keywords if kw in text)
        # Title hits weighted more
        title_lower = paper.get("title", "").lower()
        title_hits = sum(1 for kw in keywords if kw in title_lower)

        score = (hits + title_hits * 0.5) / (len(keywords) * 1.5)
        return min(1.0, score)

    # === Step 5: LLM 生成 Top-5 摘要 + 排名理由 ===

    def summarize_top5(self, papers: list[dict], query: str, search_focus: str) -> list[dict]:
        """LLM 为 Top-5 生成摘要和排名理由。"""
        if not papers:
            return []

        top5 = papers[:5]
        llm = get_llm(temperature=0.2)

        papers_text = "\n\n".join(
            f"[{i+1}] 标题: {p.get('title', '?')}\n"
            f"    作者: {', '.join(p.get('authors', [])[:3])}\n"
            f"    发表: {p.get('published', '?')} | {p.get('venue', '')}\n"
            f"    引用: {p.get('citation_count', '?')} | 来源: {p.get('source', '?')}\n"
            f"    摘要: {(p.get('abstract', '') or '')}\n"
            f"    综合评分: {p.get('composite_score', 0):.0f}/100"
            for i, p in enumerate(top5)
        )

        prompt = RANKING_PROMPT.format(
            query=query,
            search_focus=search_focus,
            papers_text=papers_text,
        )

        try:
            response = llm.invoke([
                SystemMessage(content="你是学术论文排序专家。只返回 JSON。"),
                HumanMessage(content=prompt),
            ])
            result = extract_json_from_llm_response(str(response.content))
            ranked = result.get("ranked", [])

            # 将分析结果映射回 paper
            for item in ranked:
                idx = item.get("index", 1) - 1
                if 0 <= idx < len(top5):
                    top5[idx]["core_contribution"] = item.get("core_contribution", "")
                    top5[idx]["innovation"] = item.get("innovation", "")
                    top5[idx]["method_brief"] = item.get("method_brief", "")
                    top5[idx]["relevance_reason"] = item.get("relevance_reason", "")
                    top5[idx]["ranking_reason"] = item.get("ranking_reason", "")

        except (json.JSONDecodeError, AttributeError):
            # 降级：基于摘要生成简要分析
            for p in top5:
                abstract = p.get("abstract", "") or ""
                p["core_contribution"] = abstract[:150] + ("..." if len(abstract) > 150 else "")
                p["innovation"] = ""
                p["method_brief"] = ""
                p["ranking_reason"] = f"综合评分 {p.get('composite_score', 0):.0f}/100"

        return top5

    # === 完整搜索流程 ===

    def search(self, raw_query: str, sources: list[str] | None = None) -> dict:
        """执行完整的个性化搜索流程。

        Args:
            raw_query: 用户原始查询
            sources: 搜索源列表，默认全部 (arxiv, s2, wos)

        Returns:
            {
                "papers": [Top-5 论文 + 摘要 + 理由],
                "total_found": int,
                "search_focus": str,
                "expanded_queries": list,
                "search_context": SearchContext,
            }
        """
        if sources is None:
            sources = ["arxiv", "s2"]  # wos 需要 key，默认不启用
            if os.getenv("WOS_API_KEY"):
                sources.append("wos")

        t_start = time.time()

        # Step 1: 用户画像
        ctx = self.build_search_context(raw_query)

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

        # Step 4: 多因素排序
        ranked = self.rank_papers(all_papers, raw_query, ctx)

        # Step 5: Top-5 摘要
        top5 = self.summarize_top5(ranked, raw_query, search_focus)

        duration = time.time() - t_start

        return {
            "papers": top5,
            "total_found": len(all_papers),
            "all_ranked": ranked,
            "search_focus": search_focus,
            "expanded_queries": expanded,
            "search_context": ctx,
            "duration_sec": duration,
        }
