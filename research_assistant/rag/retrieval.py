"""混合检索 — BM25 关键词 + Dense 语义向量 + RRF 融合 + LLM 重排序。

检索路径:
  1. BM25 → chunk 级关键词匹配（jieba 分词，论文元数据分句索引）
  2. Dense → Qdrant 语义向量检索（BGE 512维，句子窗口）
  3. SQLite → FTS5 全文索引直搜
  4. RRF → 三路融合排序（BM25 + Dense + SQLite 同粒度）
  5. LLM Re-rank → 对 Top-N 用 LLM 评估相关性
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

from .embedding import EmbeddingService
from .vector_store import VectorStore
from ..core.storage import PerUserStorage
from ..utils import get_llm, extract_json_from_llm_response

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """中文 jieba 分词 + 英文空格分词。"""
    tokens = []
    # 提取中文片段用 jieba 分词（保留复合词如"表面粗糙度"）
    chinese_parts = re.findall(r'[一-鿿]+', text.lower())
    if chinese_parts:
        try:
            import jieba
            jieba.setLogLevel(jieba.logging.WARNING)
            for chunk in chinese_parts:
                tokens.extend(jieba.lcut(chunk))
        except ImportError:
            for chunk in chinese_parts:
                tokens.extend(list(chunk))  # 回退逐字
    # 提取英文/数字
    english = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens.extend(english)
    return tokens or text.lower().split()


def _bm25_fingerprint(papers: list[dict]) -> str:
    """生成论文集合的内容指纹（用于检测 BM25 是否需要重建）。"""
    import hashlib
    parts = []
    for p in papers:
        pid = str(p.get("id", ""))
        ann = p.get("annotation", "")
        if isinstance(ann, dict):
            ann = json.dumps(ann, sort_keys=True, ensure_ascii=False)
        parts.append(f"{pid}:{p.get('title','')[:20]}:{ann[:100]}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


class HybridRetriever:
    """混合检索器 — BM25 + Dense + RRF + LLM re-rank。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.vector_store = VectorStore()
        self.embedder = EmbeddingService()
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[dict] = []
        self._bm25_fingerprint: str = ""  # hash 检测论文变更

    # === BM25 ===

    def ensure_bm25_index(self):
        """构建/刷新 chunk 级 BM25 索引。

        不再按论文建索引——将每篇论文的标题、摘要分句、标注关键词拆成
        多个 chunk 作为 BM25 的独立文档。这样 BM25 和 Dense 都在同一
        (chunk) 粒度，可以直接 RRF 融合，不会出现"一篇论文所有 chunk
        同幅提升"的问题。
        """
        papers = self.storage.get_all_papers()
        fp = _bm25_fingerprint(papers)
        if fp == self._bm25_fingerprint and self._bm25 is not None:
            return
        self._bm25_fingerprint = fp

        self._bm25_docs = []  # 每个元素是一个 chunk dict
        corpus = []

        for p in papers:
            pid = p.get("id", "")
            title = p.get("title", "")
            abstract = (p.get("abstract", "") or "")[:500]

            # ── 解析标注 ──
            annotation = p.get("annotation")
            if isinstance(annotation, str):
                try:
                    annotation = json.loads(annotation)
                except json.JSONDecodeError:
                    annotation = {}
            if not isinstance(annotation, dict):
                annotation = {}

            mat_kw = " ".join(annotation.get("keywords_material", []))
            meth_kw = " ".join(annotation.get("keywords_method", []))
            phen_kw = " ".join(annotation.get("keywords_phenomenon", []))
            findings = " ".join(annotation.get("key_findings", []))
            core = annotation.get("core_claim", "")

            # ── chunk 1: 标题 ──
            if title:
                corpus.append(title)
                self._bm25_docs.append({
                    "paper_id": pid, "title": title,
                    "heading_path": "Title",
                    "abstract": abstract, "core_claim": core,
                })

            # ── chunk 2..N: 摘要逐句切分 ──
            if abstract:
                import re
                sents = re.split(r'(?<=[。！？.!?])\s*', abstract)
                for sent in sents:
                    sent = sent.strip()
                    if len(sent) >= 8:
                        corpus.append(sent)
                        self._bm25_docs.append({
                            "paper_id": pid, "title": title,
                            "heading_path": "Abstract",
                            "abstract": abstract, "core_claim": core,
                        })

            # ── chunk 材料关键词 ──
            if mat_kw.strip():
                corpus.append(mat_kw)
                self._bm25_docs.append({
                    "paper_id": pid, "title": title,
                    "heading_path": "Keywords / Material",
                    "abstract": abstract, "core_claim": core,
                })

            # ── chunk 方法关键词 ──
            if meth_kw.strip():
                corpus.append(meth_kw)
                self._bm25_docs.append({
                    "paper_id": pid, "title": title,
                    "heading_path": "Keywords / Method",
                    "abstract": abstract, "core_claim": core,
                })

            # ── chunk 现象/性能关键词 ──
            if phen_kw.strip():
                corpus.append(phen_kw)
                self._bm25_docs.append({
                    "paper_id": pid, "title": title,
                    "heading_path": "Keywords / Phenomenon",
                    "abstract": abstract, "core_claim": core,
                })

            # ── chunk 关键发现 ──
            if findings.strip():
                corpus.append(findings)
                self._bm25_docs.append({
                    "paper_id": pid, "title": title,
                    "heading_path": "Key Findings",
                    "abstract": abstract, "core_claim": core,
                })

        tokenized = [_tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def bm25_search(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 chunk 级关键词搜索。

        Returns:
            chunk 级结果，每条含 paper_id / heading_path / title / bm25_score，
            可直接作为 RRF 融合的一路输入。
        """
        if self._bm25 is None:
            self.ensure_bm25_index()
        if self._bm25 is None or not self._bm25_docs:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        seen = set()
        for idx, score in indexed[:limit * 3]:  # 多取一些，后面去重
            if score <= 0:
                continue
            doc = dict(self._bm25_docs[idx])
            # 去重：同一篇论文的同一 heading_path 只留最高分
            key = f"{doc.get('paper_id', '')}|{(doc.get('heading_path', '') or '')[:60]}"
            if key in seen:
                continue
            seen.add(key)
            doc["bm25_score"] = float(score)
            doc["id"] = doc.get("paper_id")  # RRF key_fn 需要 id 字段
            results.append(doc)
        return results[:limit]

    # === Dense ===

    def dense_search(self, ctype: str, query_text: str, limit: int = 10) -> list[dict]:
        """Qdrant 语义向量检索。"""
        raw = self.vector_store.search(self.username, ctype, query_text, limit=limit)
        results = []
        for r in raw:
            payload = dict(r["payload"])  # copy
            payload["dense_score"] = r["score"]
            payload["qdrant_id"] = r["id"]
            # 映射 paper_id → id，确保 RRF 融合时与 BM25 结果的 key 一致
            # BM25 的 id 是 SQLite INTEGER，这里必须用同类型（int）否则 RRF 不会合并
            if "paper_id" in payload and "id" not in payload:
                payload["id"] = int(payload["paper_id"])
            results.append(payload)
        return results

    # === RRF 融合 ===

    @staticmethod
    def _merge_chunk_info(stored: dict, incoming: dict):
        """将 incoming 的 chunk 级信息补全到 stored 中。

        BM25 返回论文级数据(有 abstract，无 text/heading_path)
        Dense 返回 chunk 级数据(有 text/window_text/heading_path)
        合并时保留 chunk 信息，确保 LLM 看到检索命中的具体内容。
        """
        for field in ["text", "title", "heading_path", "window_text",
                       "abstract", "core_claim", "year", "venue",
                       "_original_text", "_is_window"]:
            if not stored.get(field) and incoming.get(field):
                stored[field] = incoming[field]
        # payload 也可能包含这些字段
        inc_payload = incoming.get("payload", {})
        for field in ["text", "heading_path", "window_text"]:
            if not stored.get(field) and inc_payload.get(field):
                stored[field] = inc_payload[field]

    def rrf_fuse(self, result_sets: list[list[dict]],
                 key_fn=None, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion — 融合多路排序结果。

        Args:
            result_sets: 多路结果列表
            key_fn: 提取去重 key 的函数，默认用 "id"
            k: RRF 参数

        Returns:
            融合后按 RRF 分数排序的结果
        """
        if key_fn is None:
            key_fn = lambda doc: doc.get("id") or doc.get("arxiv_id") or doc.get("title")

        rrf_scores: dict[str, tuple[float, dict]] = {}

        for results in result_sets:
            for rank, doc in enumerate(results, 1):
                key = key_fn(doc)
                rrf = 1.0 / (k + rank)
                if key in rrf_scores:
                    prev_score, prev_doc = rrf_scores[key]
                    # 合并：保留第一个文档，但用后续文档的 chunk 信息补全
                    self._merge_chunk_info(prev_doc, doc)
                    rrf_scores[key] = (prev_score + rrf, prev_doc)
                else:
                    rrf_scores[key] = (rrf, dict(doc))

        fused = []
        for key, (score, doc) in rrf_scores.items():
            doc["rrf_score"] = score
            fused.append(doc)
        fused.sort(key=lambda d: d["rrf_score"], reverse=True)
        return fused

    # === 综合检索 ===

    def search_papers(self, query: str, limit: int = 20) -> list[dict]:
        """对论文库执行混合检索。

        三路 RRF 融合（同粒度: chunk 级）:
          1. BM25 — chunk 级关键词匹配（论文元数据分句索引）
          2. Dense — Qdrant 语义向量检索（句子窗口）
          3. SQLite — FTS5 全文索引直搜
          4. heading_path 加权 — chunk 标题路径命中查询词提升

        三路结果都在 chunk 粒度，RRF key 为 (paper_id, heading_path)，
        同一篇论文不同章节的 chunk 各自独立参与排序。
        """
        # BM25 chunk 级
        bm25_results = self.bm25_search(query, limit=limit * 2)

        # Dense 向量检索
        dense_results = self.dense_search("papers", query, limit=limit)

        # 句子窗口后处理: 用 window_text 替换检索到的句子
        from .chunking import post_process_sentence_window
        dense_results = post_process_sentence_window(dense_results)

        # BM25 元的论文级字段（title/abstract/core_claim）补全到 Dense chunk
        for r in dense_results:
            pid = str(r.get("paper_id") or "")
            self._inject_bm25_meta(r, bm25_results, pid)

        # SQLite 关键词直搜
        sql_results = self.storage.search_papers_local(query.split(), limit=limit)

        # 三路 RRF 融合（BM25 + Dense + SQLite），按 (paper_id, heading_path) 去重
        fused = self.rrf_fuse(
            [bm25_results, dense_results, sql_results],
            key_fn=lambda doc: (
                f"{doc.get('paper_id') or doc.get('id')}|"
                f"{(doc.get('heading_path') or '')[:60]}"
            ),
        )

        # heading_path 关键词加权
        boosted = self._boost_by_heading_path(fused, query)

        # 按 (paper_id, heading_path) 去重：每篇论文每章节保留最佳 chunk
        deduped = self._dedup_by_paper(boosted)
        return deduped[:limit]

    @staticmethod
    def _inject_bm25_meta(chunk: dict, bm25_results: list[dict], pid: str):
        """将 BM25 结果的元数据（title/abstract/core_claim）注入 chunk。"""
        for br in bm25_results:
            bpid = str(br.get("paper_id") or br.get("id", ""))
            if bpid == pid:
                for field in ["title", "abstract", "core_claim", "year", "venue",
                              "arxiv_id", "doi", "annotation"]:
                    if not chunk.get(field) and br.get(field):
                        chunk[field] = br.get(field)
                break

    @staticmethod
    def _dedup_by_paper(results: list[dict]) -> list[dict]:
        """按 (paper_id, heading_path) 去重。

        同一篇论文的不同章节保留各自最佳 chunk，
        同一章节的多个 chunk 只保留最高分。
        """
        seen = {}
        for r in results:
            pid = r.get("paper_id") or r.get("id", "")
            hp = (r.get("heading_path") or "")[:60]  # 取前60字符分组
            key = f"{pid}|{hp}"
            if key not in seen:
                seen[key] = r
            else:
                HybridRetriever._merge_chunk_info(seen[key], r)
        # 按分数排序，但限制每篇论文最多 5 个章节结果
        deduped = list(seen.values())
        deduped.sort(
            key=lambda d: d.get("rrf_score", d.get("dense_score", 0)),
            reverse=True,
        )
        # 限制每篇论文最多出现 3 次
        paper_counts: dict[str, int] = {}
        result = []
        for d in deduped:
            pid = str(d.get("paper_id") or d.get("id", ""))
            cnt = paper_counts.get(pid, 0)
            if cnt < 3:
                result.append(d)
                paper_counts[pid] = cnt + 1
        return result

    def _boost_by_heading_path(self, results: list[dict], query: str,
                                boost_factor: float = 0.15) -> list[dict]:
        """对 heading_path 命中查询关键词的结果进行加权提升。

        原理: 如果用户搜索"研磨抛光"，chunk 的 heading_path 为
        "第四章 > 4.1 研磨工艺参数的影响"，则路径中包含了"研磨"这个关键词。
        这说明这个 chunk 直接属于用户关心的章节，应该排名更靠前。

        加权方式: rrf_score * (1 + boost_factor × 命中率)
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return results

        for r in results:
            heading = (r.get("heading_path") or "").lower()
            if not heading:
                continue
            # 计算查询词在 heading_path 中的命中率
            hits = sum(1 for t in query_terms if t in heading)
            if hits > 0:
                hit_rate = hits / len(query_terms)
                # 加权: 原有分数 × (1 + boost_factor × 命中率)
                if "rrf_score" in r:
                    r["rrf_score"] = r["rrf_score"] * (1.0 + boost_factor * hit_rate)
                if "dense_score" in r:
                    r["dense_score"] = r["dense_score"] * (1.0 + boost_factor * hit_rate)

        # 重新排序
        results.sort(key=lambda d: d.get("rrf_score", d.get("dense_score", 0)), reverse=True)
        return results

    def search_progress(self, query: str, limit: int = 10) -> list[dict]:
        """检索相关进展记录。"""
        return self.dense_search("progress", query, limit=limit)

    def search_memory(self, query: str, limit: int = 10) -> list[dict]:
        """检索相关会话记忆。"""
        return self.dense_search("memory", query, limit=limit)

    def search_all(self, query: str, intent: str = "summarize_past") -> dict[str, list[dict]]:
        """三路并行检索 + 意图自适应权重。

        Args:
            query: 用户查询
            intent: summarize_past / find_experiment / find_paper

        Returns:
            {"papers": [...], "progress": [...], "memory": [...]}
        """
        # 意图权重决定每路取多少
        if intent == "find_experiment":
            limits = {"papers": 5, "progress": 15, "memory": 5}
        elif intent == "find_paper":
            limits = {"papers": 15, "progress": 3, "memory": 3}
        else:  # summarize_past
            limits = {"papers": 10, "progress": 10, "memory": 10}

        return {
            "papers": self.search_papers(query, limit=limits["papers"]),
            "progress": self.search_progress(query, limit=limits["progress"]),
            "memory": self.search_memory(query, limit=limits["memory"]),
        }

    # === LLM Re-rank ===

    def llm_rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """用 LLM 对候选论文重排序。

        Args:
            query: 用户原始查询
            candidates: 候选论文列表（含 title, abstract, annotation 等）
            top_k: 返回数量

        Returns:
            重排序后的 Top-K 论文，每篇附带 ranking_reason
        """
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        llm = get_llm(temperature=0.1)

        # 构建论文摘要列表
        papers_text = "\n\n".join(
            f"[{i+1}] 标题: {p.get('title', '?')}\n"
            f"    摘要: {(p.get('abstract', '') or '')[:200]}\n"
            f"    年份: {p.get('year', '?')} | 引用: {p.get('citation_count', '?')}"
            for i, p in enumerate(candidates)
        )

        prompt = f"""从以下论文中选出与查询最相关的 Top-{top_k}，按相关性排序。

查询: "{query}"

{papers_text}

返回 JSON 格式: {{"ranking": [{{"index": 1, "reason": "一句话说明为什么这篇排在这"}}, ...]}}
只返回 JSON，不要任何解释。"""

        try:
            response = llm.invoke([
                SystemMessage(content="你是学术论文检索排序专家。只返回 JSON。"),
                HumanMessage(content=prompt),
            ])
            result = extract_json_from_llm_response(str(response.content))
            ranking = result.get("ranking", [])
        except Exception:
            return candidates[:top_k]

        reranked = []
        for item in ranking[:top_k]:
            idx = item["index"] - 1
            if 0 <= idx < len(candidates):
                c = dict(candidates[idx])
                c["ranking_reason"] = item.get("reason", "")
                reranked.append(c)

        # 补全数量
        if len(reranked) < top_k:
            for c in candidates:
                if c not in reranked and len(reranked) < top_k:
                    c = dict(c)
                    c["ranking_reason"] = ""
                    reranked.append(c)

        return reranked[:top_k]


    # ============================================================
    # MQE 多查询扩展 + HyDE 假设文档嵌入
    # ============================================================

    def _mqe_expand(self, query: str, n: int = 3) -> list[str]:
        """LLM 生成语义等价的多样化查询。

        参考 hello_agents 的 _prompt_mqe，适配科研场景：
        生成不同角度的检索词，覆盖同义词、上下位概念。
        """
        try:
            llm = get_llm(temperature=0.4, max_tokens=256)
            response = llm.invoke([
                SystemMessage(
                    content="你是学术检索查询扩展助手。生成语义等价或互补的多样化查询，"
                            "使用中文学术术语，简短，每行一个。"
                ),
                HumanMessage(
                    content=f"原始查询：{query}\n请给出{n}个不同表述的学术检索词，每行一个。"
                ),
            ])
            lines = []
            for ln in str(response.content).splitlines():
                # 去掉开头编号 (1. / 1、/ - / · 等)
                cleaned = ln.strip()
                cleaned = re.sub(r'^[\d]+[\.、\)\s\-\·]+', '', cleaned).strip()
                if cleaned and len(cleaned) >= 3:
                    lines.append(cleaned)
            return lines[:n] or [query]
        except Exception:
            logger.debug("MQE 扩展失败", exc_info=True)
            return [query]

    def _hyde_expand(self, query: str) -> str | None:
        """生成假设性答案段落，用于改善检索。

        HyDE 的核心前提是"生成的假设答案要和知识库同分布"。
        即使没有任何用户历史，LLM 的通用知识也能生成比原始问题
        更接近论文表述的段落。有用户画像时更精准，无画像时仍可用。
        """
        try:
            context_parts = []

            # 1. 论文关键词（厚标注中的术语）
            try:
                papers = self.storage.get_all_papers()
                if papers:
                    kws = set()
                    for p in papers[:10]:
                        ann = p.get("annotation", {})
                        if isinstance(ann, str):
                            import json
                            try:
                                ann = json.loads(ann)
                            except Exception:
                                ann = {}
                        if isinstance(ann, dict):
                            for k in ["keywords_material", "keywords_method",
                                       "keywords_phenomenon"]:
                                for w in ann.get(k, [])[:3]:
                                    if w:
                                        kws.add(w)
                    if kws:
                        context_parts.append(
                            f"知识库术语: {', '.join(list(kws)[:10])}")
            except Exception:
                pass

            # 2. 用户画像（结构化事实，来自记忆系统）
            try:
                from ..memory.user_profile import UserProfileManager
                user_dir = str(getattr(self.storage, 'user_dir', ''))
                username = getattr(self, '_username', '')
                if user_dir and username:
                    pm = UserProfileManager(username, user_dir)
                    profile_data = pm.load_facts()
                    facts = profile_data.get("facts", [])
                    if facts:
                        top_facts = sorted(
                            facts, key=lambda f: f.get("confidence", 0),
                            reverse=True)[:5]
                        fact_lines = [f["content"] for f in top_facts
                                       if f.get("content")]
                        if fact_lines:
                            context_parts.append(
                                f"用户已知信息: {'; '.join(fact_lines)}")
            except Exception:
                pass

            # 3. 进展记录
            try:
                progress = self.storage.get_all_progress(limit=3)
                if progress:
                    items = [p.get("title", "") for p in progress]
                    context_parts.append(f"研究进展: {'; '.join(items)}")
            except Exception:
                pass

            # ── 构建 prompt（有上下文更精准，无上下文仍可用）──
            user_ctx = "\n".join(context_parts) if context_parts else ""

            if user_ctx:
                system_msg = (
                    "根据问题和以下背景信息，写一段可能的答案段落。"
                    "用词与背景中的术语保持一致，用于向量检索。"
                    "直接写答案段落，不要分析过程。"
                )
                user_msg = (
                    f"背景:\n{user_ctx}\n\n"
                    f"问题: {query}\n\n"
                    f"请写一段答案段落:"
                )
            else:
                system_msg = (
                    "根据问题写一段可能的答案段落。"
                    "用于向量检索，直接写答案段落，不要分析过程。"
                )
                user_msg = f"问题: {query}\n\n请写一段答案段落:"

            llm = get_llm(temperature=0.3, max_tokens=512)
            response = llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ])
            text = str(response.content).strip()
            return text if text else None
        except Exception:
            logger.debug("HyDE 生成失败", exc_info=True)
            return None

    def search_papers_expanded(
        self,
        query: str,
        limit: int = 20,
        enable_mqe: bool = True,
        mqe_expansions: int = 3,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
    ) -> list[dict]:
        """扩展检索 — MQE + HyDE + 四路融合。

        扩展-检索-合并三步流程：
          1. 扩展：原始查询 + MQE 多样化查询 + HyDE 假设答案
          2. 检索：每个扩展查询并行执行 search_papers
          3. 合并：去重 + 按最高分排序 → top-k

        Args:
            query: 用户原始查询
            limit: 最终返回数量
            enable_mqe: 是否启用多查询扩展（默认开启）
            mqe_expansions: MQE 生成的扩展查询数
            enable_hyde: 是否启用 HyDE（默认关闭，多一次 LLM 调用）
            candidate_pool_multiplier: 候选池倍数

        Returns:
            去重合并后的 top-k 结果
        """
        # 1. 扩展查询
        expansions = [query]

        if enable_mqe:
            mqe_queries = self._mqe_expand(query, n=mqe_expansions)
            expansions.extend(mqe_queries)

        if enable_hyde:
            hyde_text = self._hyde_expand(query)
            if hyde_text:
                expansions.append(hyde_text)

        # 去重
        seen = set()
        uniq = []
        for e in expansions:
            if e and e not in seen:
                seen.add(e)
                uniq.append(e)
        expansions = uniq

        # 2. 并行检索（分配候选池）
        pool = max(limit * candidate_pool_multiplier, 20)
        per_expansion = max(1, pool // max(1, len(expansions)))

        # 用 MRR 风格的排名位置合并（不同查询的分数不可直接比较）
        rrf_agg: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        merge_k = 60

        for q in expansions:
            results = self.search_papers(q, limit=per_expansion)
            for rank, r in enumerate(results, 1):
                key = r.get("id") or r.get("arxiv_id") or r.get("title", "")
                if not key:
                    continue
                rrf = 1.0 / (merge_k + rank)
                rrf_agg[key] = rrf_agg.get(key, 0.0) + rrf
                if key not in doc_map:
                    doc_map[key] = r

        # 3. 合并排序（按累计 MRR 分数）
        merged = []
        for key, score in rrf_agg.items():
            doc = doc_map[key]
            doc["_mrr_score"] = score
            merged.append(doc)
        merged.sort(key=lambda d: d["_mrr_score"], reverse=True)

        # 按 paper_id 去重
        deduped = self._dedup_by_paper(merged)
        return deduped[:limit]
