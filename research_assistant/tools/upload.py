"""论文上传工具 — inbox 拖拽 + CLI 命令 + 文档加载解析。

上传方式:
  1. 拖拽 PDF 到 data/users/{name}/inbox/ → 自动扫描处理
  2. CLI: upload <path/to/paper.pdf>
  3. CLI: upload --dir <path/to/dir/> 批量上传
  4. CLI: inbox scan  手动扫描收件箱
  5. CLI: inbox status 查看收件箱状态

PDF 解析: MarkItDown (主力, 无GPU) → PyMuPDF (兜底, 伪MD注入)
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

from ..core.storage import PerUserStorage
from ..rag.ingestion import IngestionPipeline
from ..loaders import load_document
from ..loaders.metadata import detect_paper_type
from ..utils import get_llm, extract_json_from_llm_response

logger = logging.getLogger(__name__)


class UploadManager:
    """论文上传管理器。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.user_dir = storage.user_dir
        self.inbox_dir = self.user_dir / "inbox"
        self.processing_dir = self.user_dir / "processing"
        self.processed_dir = self.user_dir / "processed"
        self.failed_dir = self.user_dir / "failed"
        for d in [self.inbox_dir, self.processing_dir, self.processed_dir, self.failed_dir]:
            d.mkdir(exist_ok=True)
        self.pipeline = IngestionPipeline(storage, username)

    # === 类常量 ===
    MIN_TEXT_LEN = 200           # 最小有效文本长度
    ABSTRACT_MAX = 2000          # 摘要最大字符数
    LLM_FALLBACK_CHARS = 6000    # LLM 兜底最大输入字符数

    # === PDF 文本提取（委托 loaders） ===

    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, dict]:
        """统一文档加载 — MarkItDown (主力) → PyMuPDF (兜底)。

        无论期刊论文或学位论文，统一走 loaders.load_document() 流水线。
        输出为标准 Markdown，带 heading_path 信息供分块使用。

        Returns:
            (markdown_text, metadata_dict)
        """
        try:
            page_count = 0
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception:
                logger.debug("PDF 页数获取失败", exc_info=True)

            # 统一走 DocumentLoader
            markdown_text, metadata = load_document(pdf_path)

            # 如果元数据中缺少摘要，LLM 兜底
            if not metadata.get("abstract") and len(markdown_text) > self.MIN_TEXT_LEN:
                try:
                    metadata["abstract"] = self._llm_extract_abstract(
                        markdown_text[:self.LLM_FALLBACK_CHARS]
                    )
                except Exception:
                    logger.debug("LLM 摘要提取兜底失败", exc_info=True)

            return markdown_text, metadata

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"PDF 解析失败: {e}") from e

    def _llm_extract_abstract(self, text: str) -> str:
        """用 LLM 从文本开头识别摘要段落。"""
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = get_llm(temperature=0.0, max_tokens=512)
        response = llm.invoke([
            SystemMessage(content="从以下学术论文文本中提取摘要（abstract）内容。只返回摘要文本本身，不要任何解释。如果没有摘要，返回空字符串。"),
            HumanMessage(content=text),
        ])
        return str(response.content).strip()[:self.ABSTRACT_MAX]

    # === LLM 匹配卡片 ===

    def _generate_match_card(self, title: str, abstract: str, year: int) -> dict:
        """用 LLM 分析论文与用户研究画像的匹配度，生成决策卡片。

        Returns:
            {"relevance_score": 0-100, "core_contribution": "", "innovation": "",
             "relevance_reason": "", "suggested_action": ""}
        """
        if not abstract or len(abstract) < 50:
            return {"relevance_score": 0, "core_contribution": "",
                    "innovation": "", "relevance_reason": "",
                    "suggested_action": "摘要过短，建议查看原文判断"}

        # 收集用户研究上下文
        user_ctx = ""
        try:
            progress = self.storage.get_all_progress(limit=5)
            topics = set()
            for p in progress:
                t = p.get("topic", "")
                if t:
                    topics.add(t)
                insight = p.get("insights", "")
                if insight:
                    user_ctx += f"  - {insight}\n"
            if topics:
                user_ctx = f"研究主题: {', '.join(topics)}\n" + user_ctx
            # 已有论文方向
            papers = self.storage.get_all_papers()
            if papers:
                kws = set()
                for p in papers[:20]:
                    ann = p.get("annotation")
                    if isinstance(ann, str):
                        try:
                            import json
                            ann = json.loads(ann)
                        except Exception:
                            ann = {}
                    if isinstance(ann, dict):
                        for kw in ann.get("keywords_material", [])[:3]:
                            if kw:
                                kws.add(kw)
                if kws:
                    user_ctx += f"已有文献方向: {', '.join(list(kws)[:10])}\n"
        except Exception:
            logger.debug("非关键操作失败", exc_info=True)

        if not user_ctx.strip():
            user_ctx = "（暂无研究记录，需用户自行判断）"

        from langchain_core.messages import HumanMessage, SystemMessage
        llm = get_llm(temperature=0.2, max_tokens=512)

        prompt = f"""你是一个科研匹配分析助手。基于用户的研究背景，分析这篇论文与用户研究的匹配度。

## 用户研究背景
{user_ctx}

## 论文信息
标题: {title}
年份: {year}
摘要: {abstract[:1500]}

## 任务
分析这篇论文与用户研究的匹配度，返回 JSON:
{{"relevance_score": 0-100的匹配度评分,
 "core_contribution": "论文核心贡献（1句话）",
 "innovation": "创新点（1句话）",
 "relevance_reason": "与用户研究的具体关联",
 "suggested_action": "建议: 强烈推荐入库/可以入库/建议跳过/不相关"}}

只返回 JSON。"""

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研匹配分析专家。只返回 JSON。"),
                HumanMessage(content=prompt),
            ])
            content = str(response.content)
            return extract_json_from_llm_response(content)
        except Exception:
            return {"relevance_score": 0, "core_contribution": "（分析失败）",
                    "innovation": "", "relevance_reason": "",
                    "suggested_action": "请手动判断"}

    def _check_incomplete(self, paper_id: int) -> bool:
        """检查论文在 Qdrant 中是否有 chunk 数据。

        返回 True 表示入库不完整（Qdrant 无 chunk），需要重新入库。
        """
        try:
            from qdrant_client import QdrantClient
            import os
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=url)
            collection = f"user_{self.username}_papers"
            # 用 paper_id filter scroll 几条，确认有数据
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter={"must": [{"key": "paper_id", "match": {"value": paper_id}}]},
                limit=1,
            )
            if points:
                return False  # 有数据，完整
            # 再试不带 filter（Qdrant 版本兼容）
            points, _ = client.scroll(collection_name=collection, limit=100)
            for pt in points:
                payload = pt.payload or {}
                if payload.get("paper_id") == paper_id:
                    return False
            return True  # 无数据，不完整
        except Exception:
            # Qdrant 挂了也算不完整，让用户之后重试
            return True

    # === 单文件上传 ===

    def upload_file(self, file_path: str, confirm: bool = True) -> dict:
        """上传单个 PDF 文件。

        Args:
            file_path: PDF 文件路径
            confirm: 是否确认入库（False 则自动入库）

        Returns:
            {"status": "ok"/"error"/"skip", "message": "...", "paper_id": int|None}
        """
        file_path = str(file_path)
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"文件不存在: {file_path}"}

        if not file_path.lower().endswith(".pdf"):
            return {"status": "error", "message": "只支持 PDF 文件"}

        fname = os.path.basename(file_path)

        # 提取文本
        try:
            full_text, pdf_meta = self.extract_text_from_pdf(file_path)
        except Exception as e:
            return {"status": "error", "message": f"PDF 解析失败: {e}"}

        if len(full_text) < self.MIN_TEXT_LEN:
            return {
                "status": "error",
                "message": (
                    f"PDF 可提取文本仅 {len(full_text)} 字符，可能是扫描版或图片 PDF。\n"
                    "当前系统不支持 OCR，建议:\n"
                    "  1) 使用 Adobe Acrobat 的 OCR 功能先转换\n"
                    "  2) 或寻找该论文的文本版 PDF"
                ),
            }

        # 合并元数据
        paper = {
            "title": pdf_meta.get("title", fname.replace(".pdf", "")),
            "authors": pdf_meta.get("authors", []),
            "abstract": pdf_meta.get("abstract", ""),
            "year": datetime.now().year,
            "source": "user_upload",
            "file_path": file_path,
            "pdf_url": "",
            "citation_count": 0,
            "venue": "",
        }

        ext_method = pdf_meta.get("extraction_method", "unknown")
        ext_label = "[Marker 解析]" if ext_method == "marker" else "[PyMuPDF 降级]"

        # 生成 LLM 匹配卡片
        card = self._generate_match_card(
            paper["title"],
            paper["abstract"],
            paper["year"],
        )

        # 预览
        preview = {
            "title": paper["title"],
            "pages": pdf_meta.get("pages", 0),
            "abstract": paper["abstract"] if paper["abstract"] else "(未提取到摘要)",
            "text_length": len(full_text),
            "extraction_method": ext_method,
            "card": card,
        }

        if confirm:
            score = card.get("relevance_score", 0)
            print(f"\n  {'='*50}")
            print(f"  标题: {paper['title'][:80]}")
            print(f"  页数: {pdf_meta.get('pages', '?')}  {ext_label}")
            print(f"  {'='*50}")
            print(f"  匹配度: {score}/100")
            if card.get("core_contribution"):
                print(f"  核心贡献: {card['core_contribution']}")
            if card.get("innovation"):
                print(f"  创新点: {card['innovation']}")
            if card.get("relevance_reason"):
                print(f"  与你的关联: {card['relevance_reason']}")
            if card.get("suggested_action"):
                print(f"  建议: {card['suggested_action']}")
            if paper["abstract"]:
                print(f"\n  原文摘要 ({len(paper['abstract'])} 字符):")
                print(f"  {paper['abstract'][:400]}{'...' if len(paper['abstract'])>400 else ''}")
            print()
            user_input = input("  确认入库？(Y/n): ").strip().lower()
            if user_input and user_input != 'y':
                return {"status": "skip", "message": "已跳过", "preview": preview}

        # 去重检查（同时验证已有论文是否完整入库）
        existing = self.storage.get_all_papers()
        for ep in existing:
            if ep.get("title", "").strip() == paper.get("title", "").strip():
                existing_id = ep["id"]
                incomplete = self._check_incomplete(existing_id)
                if incomplete:
                    print(f"\n  ⚠ 已存在同名论文 (ID={existing_id})，但 Qdrant 无 chunk 数据。")
                    print(f"    可能上次入库时嵌入步骤超时，将删除不完整记录并重新入库。")
                    self.storage.delete_paper(existing_id)
                    break  # 跳出循环，继续正常入库流程
                else:
                    print(f"\n  ⚠ 已存在同名论文 (ID={ep['id']})，自动跳过。")
                    return {"status": "skip", "message": f"已存在 (ID={ep['id']})",
                            "paper_id": ep['id'], "preview": preview}

        # 入库
        try:
            paper_id = self.pipeline.ingest(paper, full_text)
            return {
                "status": "ok",
                "message": f"已入库: {paper['title'][:60]}",
                "paper_id": paper_id,
                "preview": preview,
            }
        except Exception as e:
            return {"status": "error", "message": f"入库失败: {e}", "preview": preview}

    # === Inbox 扫描 ===

    def scan_inbox(self) -> list[dict]:
        """扫描 inbox 目录，处理所有待上传的 PDF。

        Returns:
            [{status, message, paper_id, file_name}, ...]
        """
        pdf_files = list(self.inbox_dir.glob("*.pdf"))
        if not pdf_files:
            return []

        results = []
        for pdf_path in pdf_files:
            # 移动到 processing
            dest = self.processing_dir / pdf_path.name
            shutil.move(str(pdf_path), str(dest))

            try:
                result = self.upload_file(str(dest), confirm=False)
                result["file_name"] = pdf_path.name

                if result["status"] == "ok":
                    # 成功 → 移到 processed
                    shutil.move(str(dest), str(self.processed_dir / pdf_path.name))
                else:
                    # 失败 → 移到 failed
                    shutil.move(str(dest), str(self.failed_dir / pdf_path.name))
                    # 写错误日志
                    error_log = self.failed_dir / f"{pdf_path.stem}.error.log"
                    error_log.write_text(
                        f"上传时间: {datetime.now().isoformat()}\n"
                        f"错误: {result.get('message', 'Unknown')}\n",
                        encoding="utf-8",
                    )

                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "message": str(e),
                    "paper_id": None,
                    "file_name": pdf_path.name,
                })
                # 移到 failed
                try:
                    shutil.move(str(dest), str(self.failed_dir / pdf_path.name))
                except Exception:
                    pass

        return results

    def inbox_status(self) -> dict:
        """查看收件箱状态。"""
        pending = list(self.inbox_dir.glob("*.pdf"))
        processing = list(self.processing_dir.glob("*.pdf"))
        processed = list(self.processed_dir.glob("*.pdf"))
        failed = list(self.failed_dir.glob("*.pdf"))

        return {
            "pending": len(pending),
            "pending_files": [f.name for f in pending],
            "processing": len(processing),
            "processed": len(processed),
            "failed": len(failed),
            "failed_files": [f.name for f in failed],
        }

    def upload_directory(self, dir_path: str) -> list[dict]:
        """批量上传一个目录中的所有 PDF。"""
        d = Path(dir_path)
        if not d.is_dir():
            return [{"status": "error", "message": f"不是目录: {dir_path}"}]

        results = []
        for pdf_path in d.glob("*.pdf"):
            result = self.upload_file(str(pdf_path), confirm=False)
            result["file_name"] = pdf_path.name
            results.append(result)
        return results
