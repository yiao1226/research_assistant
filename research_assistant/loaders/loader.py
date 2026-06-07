"""DocumentLoader — 统一文档加载入口。

策略（按文件类型）:
  PDF:      PyMuPDF 提取 → 伪MD注入 → Markdown
  Word/Excel/PPT/图片/HTML 等: MarkItDown 转换 → Markdown
  纯文本:  直接读取

经过中英论文全篇实测验证:
  - MarkItDown 对双栏期刊 PDF 产出损坏文本（单词空格全部丢失）
  - MarkItDown 对中文学位论文结构差，14秒产出不如 PyMuPDF 0.4秒
  - PyMuPDF 对两类论文均稳定产出干净结构和标题
  → PDF 走 PyMuPDF，非 PDF 走 MarkItDown
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MIN_TEXT_LEN = 200
LLM_FALLBACK_CHARS = 6000

# MarkItDown 支持的非 PDF 格式
_MARKITDOWN_FORMATS = {
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp',
    '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg',
    '.zip', '.tar', '.gz',
}


class DocumentLoader:
    """统一文档加载器。根据文件类型自动选择最优加载器。"""

    def load(self, path: str) -> tuple[str, dict]:
        """加载任意文档，返回 (markdown_text, metadata)。

        PDF → PyMuPDF + 伪MD注入
        Word/Excel/图片等 → MarkItDown
        纯文本 → 直接读取

        Raises:
            FileNotFoundError, ValueError
        """
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = (os.path.splitext(path)[1] or '').lower()

        if ext == '.pdf':
            return self._load_pdf(path)
        elif ext in _MARKITDOWN_FORMATS:
            return self._load_with_markitdown(path)
        else:
            return self._load_text(path)

    # ── PDF: PyMuPDF ──

    def _load_pdf(self, pdf_path: str) -> tuple[str, dict]:
        """PyMuPDF 主力 PDF 加载。"""
        from .pymupdf_loader import extract_text_pymupdf

        markdown_text, metadata = extract_text_pymupdf(pdf_path)

        if not markdown_text or len(markdown_text) < MIN_TEXT_LEN:
            raise ValueError(
                f"PDF 可提取文本仅 {len(markdown_text)} 字符，"
                "可能是扫描版。建议先用 OCR 转换。"
            )

        # 摘要 LLM 兜底
        if not metadata.get("abstract") and len(markdown_text) > MIN_TEXT_LEN:
            try:
                metadata["abstract"] = _llm_extract_abstract(
                    markdown_text[:LLM_FALLBACK_CHARS]
                )
            except Exception:
                logger.debug("LLM 摘要兜底失败", exc_info=True)

        return markdown_text, metadata

    # ── 非 PDF: MarkItDown ──

    def _load_with_markitdown(self, path: str) -> tuple[str, dict]:
        """MarkItDown 处理 Word/Excel/图片等非 PDF 格式。"""
        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ImportError("MarkItDown 未安装，无法处理此格式: pip install markitdown[all]")

        md = MarkItDown()
        result = md.convert(path)
        text = str(result.text_content)

        if not text.strip():
            raise ValueError(f"MarkItDown 无法从此文件提取内容: {path}")

        metadata = {
            "title": "", "authors": [], "abstract": "",
            "file_path": path,
            "extraction_method": "markitdown",
        }
        return text, metadata

    # ── 纯文本 ──

    def _load_text(self, path: str) -> tuple[str, dict]:
        """直接读取纯文本文件。"""
        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法解码文件: {path}")

        metadata = {
            "title": os.path.basename(path),
            "authors": [], "abstract": "",
            "file_path": path,
            "extraction_method": "text",
        }
        return text, metadata


# ── 便利函数 ──

def load_document(path: str) -> tuple[str, dict]:
    """加载任意文档。"""
    return DocumentLoader().load(path)


def _llm_extract_abstract(text: str) -> str:
    """LLM 兜底摘要提取。"""
    from ..utils import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(temperature=0.0, max_tokens=512)
    response = llm.invoke([
        SystemMessage(content="从以下学术论文文本中提取摘要。只返回摘要本身，不要任何解释。如果没有摘要，返回空字符串。"),
        HumanMessage(content=text),
    ])
    return str(response.content).strip()[:2000]
