"""Qdrant 向量存储封装 — 每个用户独立的 collection 体系。

Collection 命名: user_{username}_{collection_type}
  - papers:    论文（按章节 chunk）
  - progress:  用户进展记录
  - memory:    会话摘要
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .embedding import EmbeddingService


class VectorStore:
    """Qdrant 向量存储，单例（全局一个 client，每个用户独立 collection）。"""

    _instance: Optional["VectorStore"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY", "") or None
        if qdrant_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        else:
            self.client = QdrantClient(url=qdrant_url)
        self.embedder = EmbeddingService()

    # === Collection 管理 ===

    def _collection_name(self, username: str, ctype: str) -> str:
        """user_{username}_{ctype}"""
        return f"user_{username}_{ctype}"

    def ensure_collection(self, username: str, ctype: str):
        """确保 collection 存在。"""
        name = self._collection_name(username, ctype)
        # 先列出现有 collection（网络错误直接上抛，不吞异常）
        existing = {c.name for c in self.client.get_collections().collections}
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedder.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def ensure_all_collections(self, username: str):
        """为某用户创建所有 collection。"""
        for ctype in ["papers", "progress", "memory"]:
            self.ensure_collection(username, ctype)

    # === Upsert（插入或更新） ===

    EMBED_BATCH = 256  # 每批编码数量，避免 CPU OOM 或超时

    def upsert(self, username: str, ctype: str, points: list[dict]):
        """批量插入向量（分批编码 + 分批写入，带进度提示）。

        Args:
            username: 用户名
            ctype: collection 类型 (papers/progress/memory)
            points: [{"id": str, "text": str, "payload": dict}, ...]
                    其中 "text" 是要嵌入的文本，嵌入后会替换为 "vector"
        """
        self.ensure_collection(username, ctype)
        name = self._collection_name(username, ctype)

        total = len(points)
        if total == 0:
            return

        if total == 1:
            print(f"  [Embed] 编码 1 个文本块...", flush=True)
            vecs = [self.embedder.encode_document(points[0]["text"])]
            qpoints = [self._make_qpoint(points[0], vecs[0])]
            self.client.upsert(collection_name=name, points=qpoints)
            return

        # 大批量：分批编码 + 分批写入 Qdrant
        batch_count = (total + self.EMBED_BATCH - 1) // self.EMBED_BATCH
        print(f"  [Embed] 共 {total} 个文本块，分 {batch_count} 批编码...", flush=True)

        for b in range(batch_count):
            start = b * self.EMBED_BATCH
            end = min(start + self.EMBED_BATCH, total)
            batch_points = points[start:end]
            texts = [p["text"] for p in batch_points]

            print(f"  [Embed] 批次 {b+1}/{batch_count} ({start+1}-{end}/{total})...",
                  flush=True)
            vecs = self.embedder.encode(texts)

            qpoints = [self._make_qpoint(pt, vec)
                       for pt, vec in zip(batch_points, vecs)]
            self.client.upsert(collection_name=name, points=qpoints)

        print(f"  [Embed] 全部 {total} 个文本块编码完成", flush=True)

    def _make_qpoint(self, point: dict, vec: list[float]):
        pid = point.get("id", str(uuid.uuid4()))
        payload = point.get("payload", {})
        payload["text"] = point["text"]
        return qmodels.PointStruct(id=pid, vector=vec, payload=payload)

    # === Search ===

    def search(self, username: str, ctype: str, query_text: str,
               limit: int = 10, score_threshold: float = 0.0) -> list[dict]:
        """向量相似度搜索。

        Returns:
            [{"id": str, "score": float, "payload": dict}, ...]
        """
        self.ensure_collection(username, ctype)
        name = self._collection_name(username, ctype)
        query_vec = self.embedder.encode_query(query_text)

        results = self.client.query_points(
            collection_name=name,
            query=query_vec,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload or {}}
            for r in results.points
        ]

    def search_batch(self, username: str, queries: list[dict]) -> list[list[dict]]:
        """批量搜索多个 collection。

        Args:
            queries: [{"ctype": "papers", "text": "...", "limit": 5}, ...]

        Returns:
            [[result, ...], ...]
        """
        results = []
        for q in queries:
            r = self.search(
                username=username,
                ctype=q["ctype"],
                query_text=q["text"],
                limit=q.get("limit", 10),
            )
            results.append(r)
        return results

    # === Delete ===

    def delete_points(self, username: str, ctype: str, point_ids: list[str]):
        name = self._collection_name(username, ctype)
        try:
            self.client.delete(collection_name=name, points_selector=point_ids)
        except Exception:
            pass

    def delete_by_paper_id(self, username: str, ctype: str, paper_id: int):
        """删除指定论文的所有 chunk 向量。使用 Qdrant 的 payload 过滤删除。"""
        name = self._collection_name(username, ctype)
        try:
            self.client.delete(
                collection_name=name,
                points_selector=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="paper_id",
                        match=qmodels.MatchValue(value=paper_id),
                    )]
                ),
            )
        except Exception:
            pass

    def delete_collection(self, username: str, ctype: str):
        name = self._collection_name(username, ctype)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
