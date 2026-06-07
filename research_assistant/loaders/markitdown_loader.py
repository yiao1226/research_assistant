"""MarkItDown 加载器 — 非 PDF 格式→Markdown 转换。

用于 Word/Excel/PPT/图片/HTML 等格式。PDF 格式走 pymupdf_loader.py。
"""
from __future__ import annotations

import logging

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

logger = logging.getLogger(__name__)


def extract_text_markitdown(path: str) -> tuple[str, dict]:
    """使用 MarkItDown 将任意格式文件转换为 Markdown。

    适用于: .docx, .xlsx, .pptx, .html, .jpg, .png, .mp3 等。
    不适用于 PDF（走 pymupdf_loader.py）。

    Args:
        path: 文件路径

    Returns:
        (markdown_text, metadata_dict)

    Raises:
        ImportError: markitdown 未安装
        Exception: 转换失败
    """
    if MarkItDown is None:
        raise ImportError("markitdown 未安装，请执行 pip install markitdown")
    md = MarkItDown()
    result = md.convert(path)
    text = str(result.text_content)

    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "pages": 0,
        "file_path": path,
        "extraction_method": "markitdown",
    }

    return text, metadata
