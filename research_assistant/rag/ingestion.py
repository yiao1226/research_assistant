"""论文入库管线 — 厚标注 + 分章 Chunk + 嵌入 + 双写（SQLite + Qdrant）。

入库流程:
  1. 接收论文 dict（来自搜索 API 或上传 PDF）
  2. LLM 厚标注（一次性，~800-2600 token）
  3. 标注完整性验证
  4. 分章节 chunk（如有全文）或单 chunk（只有摘要）
  5. BGE 嵌入 + 写入 Qdrant
  6. 元数据 + 标注写入 SQLite
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.storage import PerUserStorage
from .vector_store import VectorStore
from ..utils import get_llm, extract_json_from_llm_response
from ..schemas import PaperAnnotation

logger = logging.getLogger(__name__)


def _is_zh_text(sample: str) -> bool:
    """检测文本是否以中文为主。"""
    cjk = sum(1 for c in sample if '一' <= c <= '鿿')
    ascii_alpha = sum(1 for c in sample if c.isascii() and c.isalpha())
    return cjk > ascii_alpha


ANNOTATION_SYSTEM_PROMPT = """你是科研论文标注专家。从论文内容中提取全面的结构化标注。

## 要求
- 不要省略任何可能被后续检索用到的关键词
- 材料名称同时保留中英文
- 方法要具体到技术细节（不仅"表征"，而是"PLQY稳态光致发光光谱"）
- 定量结果保留具体数值
- 缩写同时保留全称和缩写
- 这是一次性的标注，后续所有检索依赖这些标签的完整度

## 输出 JSON 格式
{
  "keywords_material": ["材料/化合物名称列表，中英双语"],
  "keywords_method": ["方法/技术/工艺名称列表"],
  "keywords_phenomenon": ["现象/指标/性能名称列表"],
  "methods": [
    {
      "method_name": "具体方法名称",
      "method_category": "fabrication/characterization/simulation/analysis",
      "key_parameters": {"参数名": "参数值"},
      "what_it_measures": "该方法用来测量/证明什么",
      "equipment": "关键设备（如有）"
    }
  ],
  "key_findings": ["定量或定性发现列表，保留具体数值"],
  "contribution_type": "review/experiment/theory/benchmark",
  "core_claim": "一句话核心结论（中文，<50字）",
  "solved_problem": "解决了什么问题",
  "remaining_gap": "遗留了什么缺口（如有）",
  "baseline_methods": ["论文对比的基线方法"],
  "improvement_over_baseline": "相对基线的提升（如有，保留数值）"
}

只返回 JSON，不要任何解释。"""


class IngestionPipeline:
    """论文入库管线。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.vector_store = VectorStore()

    # === 标注生成 ===

    def generate_annotation(self, paper: dict, full_text: str = "") -> dict:
        """对单篇论文生成厚标注。

        Args:
            paper: 论文元数据（title, authors, abstract, ...）
            full_text: 全文内容（如有），为空则仅基于摘要标注

        Returns:
            annotation dict
        """
        llm = get_llm(temperature=0.1)

        if full_text:
            # 截取关键章节
            text_input = self._extract_key_sections(full_text, max_chars=2500)
            quality = "full_text"
        else:
            text_input = f"标题: {paper.get('title', '')}\n摘要: {paper.get('abstract', '')}"
            quality = "abstract_only"

        if not text_input.strip():
            return {"error": "无可用文本", "quality": "none"}

        try:
            structured_llm = llm.with_structured_output(PaperAnnotation)
            result = structured_llm.invoke([
                SystemMessage(content=ANNOTATION_SYSTEM_PROMPT),
                HumanMessage(content=text_input),
            ])
            annotation = result.model_dump()
            annotation["annotation_quality"] = quality
            # methods 从 Pydantic model 序列化为 list[dict]
            return annotation
        except Exception:
            return {
                "keywords_material": [],
                "keywords_method": [],
                "keywords_phenomenon": [],
                "methods": [],
                "key_findings": [],
                "contribution_type": "unknown",
                "core_claim": "",
                "solved_problem": "",
                "remaining_gap": "",
                "baseline_methods": [],
                "improvement_over_baseline": "",
                "annotation_quality": quality,
                "parse_error": True,
            }

    def validate_annotation(self, annotation: dict) -> list[str]:
        """验证标注完整度，返回不足项列表。"""
        flags = []
        if len(annotation.get("methods", [])) < 2:
            flags.append("方法提取不足（<2）")
        if len(annotation.get("key_findings", [])) < 3:
            flags.append("结论提取不足（<3）")
        if len(annotation.get("keywords_material", [])) < 3:
            flags.append("材料关键词不足（<3）")
        if not annotation.get("core_claim"):
            flags.append("缺少核心结论")
        return flags

    # === 嵌入文本构建 ===

    def build_embedding_text(self, paper: dict, annotation: dict) -> str:
        """构建嵌入用的聚合文本——标注关键词混入提升召回命中率。"""
        parts = [
            paper.get("title", ""),
            paper.get("abstract", "") or "",
        ]

        if annotation:
            for key in ["keywords_material", "keywords_method",
                        "keywords_phenomenon", "key_findings"]:
                vals = annotation.get(key, [])
                if isinstance(vals, list):
                    parts.append(" ".join(vals))
            for key in ["core_claim", "solved_problem", "remaining_gap",
                        "contribution_type"]:
                val = annotation.get(key, "")
                if val:
                    parts.append(str(val))

        return " ".join(p for p in parts if p)

    # === 分章节 Chunk（委托给 chunking 模块） ===

    def chunk_paper(self, full_text: str, paper_id: int, annotation: dict) -> list[dict]:
        """将论文全文按句子窗口策略分块（委托 chunking.py）。

        句子窗口策略: 单个句子向量检索 + 前后各3句上下文窗口。
        检索精准（句子粒度）+ 生成上下文充分（宽窗口）。

        Returns:
            [{"id": uuid5, "text": "单句",
              "payload": {"window_text": "前后N句上下文", "heading_path", ...}}, ...]
        """
        from .chunking import chunk_paper_sentence_window as _chunk
        if not full_text:
            return []
        # 中文学位论文句子短(24t), window_size=5 窗口≈250t
        # 英文期刊句子长(70t), window_size=2 窗口≈350t
        window = 5 if _is_zh_text(full_text[:5000]) else 2
        return _chunk(full_text, paper_id, annotation, window_size=window)

    def chunk_summary_only(self, paper: dict, paper_id: int, annotation: dict) -> list[dict]:
        """仅有摘要时的单 chunk 策略。payload 结构与 chunk_paper 统一。"""
        text = self.build_embedding_text(paper, annotation)
        return [{
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"paper_{paper_id}_summary")),
            "text": text,
            "payload": {
                "paper_id": paper_id,
                "heading_path": "Abstract",
                "chunk_index": 0,
                "chunk_count": 1,
                "contribution_type": annotation.get("contribution_type", ""),
                "core_claim": annotation.get("core_claim", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "is_reference": False,
                "char_start": 0,
                "char_end": len(text),
            },
        }]

    # === 完整入库 ===

    def ingest(self, paper: dict, full_text: str = "") -> int:
        """完整入库流程: 标注 → 验证 → Chunk → Embed → 双写。

        Returns:
            paper_id (SQLite)
        """
        # 1. 标注
        annotation = self.generate_annotation(paper, full_text)
        quality = annotation.get("annotation_quality", "abstract_only")

        # 2. 验证 + 不足补全
        flags = self.validate_annotation(annotation)
        if flags and quality == "full_text":
            # 二次补全
            supplement = self._supplement_annotation(paper, full_text, flags)
            if supplement:
                annotation.update(supplement)
                # 重验
                flags = self.validate_annotation(annotation)
                if flags:
                    annotation["low_quality"] = True

        paper["annotation"] = annotation
        paper["annotation_quality"] = quality

        # 3. 写入 SQLite（先拿到 paper_id 才能做 chunk）
        paper_id = self.storage.add_paper(paper)

        # 4. Chunk + 嵌入 + 写入 Qdrant
        #    如果此步失败，回滚 SQLite 记录，避免"半入库"状态
        try:
            if full_text:
                chunks = self.chunk_paper(full_text, paper_id, annotation)
            else:
                chunks = self.chunk_summary_only(paper, paper_id, annotation)

            if chunks:
                self.vector_store.upsert(self.username, "papers", chunks)
        except Exception:
            logger.warning("嵌入/Qdrant写入失败，回滚 SQLite 记录 paper_id=%d", paper_id)
            self.storage.delete_paper(paper_id)
            raise

        # 5. 语义记忆提取（Neo4j 知识图谱）
        if full_text and annotation:
            try:
                from ..memory.semantic import SemanticMemory
                heading_paths = [
                    c["payload"].get("heading_path", "")
                    for c in chunks if c.get("payload", {}).get("heading_path")
                ]
                sm = SemanticMemory(self.username)
                if sm.available:
                    result = sm.extract_and_store(
                        paper_id=paper_id,
                        annotation=annotation,
                        heading_paths=heading_paths,
                        title=paper.get("title", ""),
                    )
                    logger.info(
                        "语义记忆: 论文 %d 抽取 %d 实体, %d 关系",
                        paper_id,
                        result.get("entity_count", 0),
                        result.get("relation_count", 0),
                    )
                sm.close()
            except Exception:
                logger.debug("语义记忆提取失败（Neo4j 离线或 LLM 异常）", exc_info=True)

        return paper_id

    # === Helpers ===

    def _extract_key_sections(self, full_text: str, max_chars: int = 2500) -> str:
        """从全文提取关键章节（Abstract + Introduction 首段 + 每节首段 + Conclusion）。

        使用 chunking.split_md_paragraphs 获取标题感知段落。
        """
        from .chunking import split_md_paragraphs
        key_order = ["abstract", "introduction", "method", "results",
                     "discussion", "conclusion", "摘要", "引言", "方法",
                     "实验", "结果", "讨论", "结论"]
        paragraphs = split_md_paragraphs(full_text)
        result = []
        seen = set()
        total = 0
        for key in key_order:
            for p in paragraphs:
                heading = p.get("heading_path", "").lower()
                content_hash = hash(p["content"])
                if key in heading and content_hash not in seen:
                    seen.add(content_hash)
                    excerpt = p["content"][:800]
                    result.append(excerpt)
                    total += len(excerpt)
                    if total >= max_chars:
                        break
            if total >= max_chars:
                break
        return "\n\n".join(result)

    def _supplement_annotation(self, paper: dict, full_text: str,
                               flags: list[str]) -> dict | None:
        """标注不足时二次补全。"""
        llm = get_llm(temperature=0.05)
        text_input = self._extract_key_sections(full_text, max_chars=2000)
        issues = "\n".join(f"- {f}" for f in flags)

        supplement_prompt = f"""之前的标注遗漏了以下内容，请补全:
{issues}

论文内容:
{text_input}

请补充缺失的字段，以 JSON 格式返回（只返回要补充的字段，不需要重复已有的）。
只返回 JSON。"""

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研标注专家。只返回 JSON。"),
                HumanMessage(content=supplement_prompt),
            ])
            content = str(response.content)
            return extract_json_from_llm_response(content)
        except Exception:
            return None
