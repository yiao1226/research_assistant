"""元数据提取 — 从 Markdown 或 PyMuPDF 文本中提取标题/作者/摘要/论文类型。

从 upload.py 中抽出，被 markitdown_loader 和 pymupdf_loader 共用。
"""
from __future__ import annotations

import re
from typing import Optional


def extract_title_from_md(markdown_text: str) -> str:
    """从 Markdown 文本提取标题：第一个 #/## 行。"""
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if re.match(r'^#{1,2}\s+', stripped) and not stripped.startswith("!["):
            title = re.sub(r'^#{1,2}\s+', '', stripped).strip()
            if len(title) > 5 and not re.match(r'^[\d\s\.\-]+$', title):
                return title
    return ""


def extract_abstract_from_md(markdown_text: str) -> str:
    """从 Markdown 文本提取摘要：## Abstract/摘要 后的段落。"""
    lines = markdown_text.split("\n")
    in_abstract = False
    abstract_lines = []

    for line in lines:
        stripped = line.strip()
        if re.match(r'^#{1,3}\s*(abstract|摘要)', stripped, re.IGNORECASE):
            in_abstract = True
        elif in_abstract:
            if stripped.startswith("#"):
                break
            if stripped:
                abstract_lines.append(stripped)

    if abstract_lines:
        return " ".join(abstract_lines)[:2000]
    return ""


def detect_paper_type(full_text: str, page_count: int) -> tuple[str, str]:
    """检测论文类型和语言。

    Returns:
        (paper_type, language): ("thesis"|"journal", "zh"|"en"|"bilingual")
    """
    chinese_chars = len(re.findall(r'[一-鿿]', full_text[:10000]))
    english_words = len(re.findall(r'[a-zA-Z]{3,}', full_text[:10000]))
    if chinese_chars > 200:
        language = "bilingual" if english_words > 500 else "zh"
    else:
        language = "en"

    THESIS_PAGE_THRESHOLD = 40
    thesis_indicators = [
        "目录", "Table of Contents", "致谢", "Acknowledgement",
        "参考文献", "References", "学位论文", "Dissertation",
        "第1章", "Chapter 1", "第一章",
    ]
    thesis_score = sum(
        1 for ind in thesis_indicators if ind.lower() in full_text[:20000].lower()
    )
    is_thesis = (
        page_count > THESIS_PAGE_THRESHOLD and thesis_score >= 2
    ) or (page_count > THESIS_PAGE_THRESHOLD * 2 and thesis_score >= 1)

    return ("thesis" if is_thesis else "journal", language)
