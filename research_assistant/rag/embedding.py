"""BGE 嵌入服务 — 中英文双语语义向量。

模型: BAAI/bge-small-zh-v1.5
特点:
  - 中英双语，512 维向量
  - 语义理解优于单纯关键词匹配
  - 支持 instruction prefix 提升检索效果
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """BGE 嵌入模型服务，单例模式避免重复加载。"""

    _instance: Optional["EmbeddingService"] = None
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
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        # 配置 HF 镜像（国内用户需要）
        hf_endpoint = os.getenv("HF_ENDPOINT", "")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        # 先尝试离线加载（已缓存则秒开），失败再在线加载
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                raise RuntimeError(
                    f"无法加载嵌入模型 '{model_name}'。\n"
                    f"  离线加载失败，在线加载也失败。\n"
                    f"  如果在国内，请在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com\n"
                    f"  原始错误: {e}"
                )

        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], instruction: str = "") -> list[list[float]]:
        """批量编码文本为向量。

        Args:
            texts: 文本列表
            instruction: BGE 的指令前缀（检索时用 "为这个句子生成表示以用于检索相关文章："）

        Returns:
            [[float, ...], ...] 向量列表
        """
        if instruction:
            texts = [instruction + t for t in texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """编码查询文本（自动加 instruction prefix）。"""
        embeddings = self.encode(
            [query],
            instruction="为这个句子生成表示以用于检索相关文章：",
        )
        return embeddings[0]

    def encode_document(self, text: str) -> list[float]:
        """编码文档文本。"""
        embeddings = self.encode([text])
        return embeddings[0]
