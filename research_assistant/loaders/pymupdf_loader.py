"""PyMuPDF 兜底加载器 — 纯文本提取 + 伪 Markdown 注入 + 页眉页脚清洗。

当 MarkItDown 加载 PDF 失败时使用此兜底方案。
输出标准 Markdown 格式（注入 # 章节标题），可供 chunking.py 直接处理。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import fitz  # PyMuPDF


HEADING_PATTERNS: list[tuple[str, str]] = [
    # 中文一级：第一章、第二章... → ## 第一章 XXX
    (r'^(第[一二三四五六七八九十\d]+章)\s*(.*)', r'## \1 \2'),
    # 中文二级：1.1 X射线... → ### 1.1 X射线...
    (r'^(\d+\.\d+(?:\.\d+)?)\s+([^\d].{3,})$', r'### \1 \2'),
    # 中文二级变体：2.1 溶液法...
    (r'^(\d+\.\d+(?:\.\d+)?)$', r'### \1'),
    # 英文一级：1. Introduction → ## 1. Introduction (至少4个字母，排除纯数字+符号)
    (r'^(\d+)\.\s+([A-Z][a-zA-Z\s&]{4,})$', r'## \1. \2'),
    # 英文二级：2.1 Sample → ### 2.1 Sample
    (r'^(\d+\.\d+)\s+([A-Z][a-zA-Z\s&]{3,})$', r'### \1 \2'),
    # 无编号一级关键词（英文）
    (r'^(Abstract|Introduction|Related\s*Work|Method|Experiment|Result|Discussion|Conclusion|References?|Acknowledgments?)$',
     r'## \1'),
    # 无编号中英文
    (r'^(摘要|Abstract|引言|方法|实验|结果|讨论|结论|参考文献|致谢|目录|附录|总结与展望)$',
     r'## \1'),
    # §2.1 等特殊格式
    (r'^(§\d+(?:\.\d+)*)\s+(.{3,})$', r'### \1 \2'),
]

# 页眉页脚高频行关键词（补充频率统计不够的情况）
FOOTER_NOISE = [
    "www.advancedsciencenews.com",
    "wileyonlinelibrary.com",
    "Adv. Funct. Mater.",
    "Adv. Mater.",
    "Supporting Information",
    "Electronic Supplementary Material",
    "This journal is",
    "All rights reserved",
    "View Article Online",
    "Published on",
    "DOI:",
]


def _clean_page_headers(text: str) -> str:
    """统计每行出现频率，高频行（>总行数×0.25）或匹配已知页脚模式 视为页眉/页脚剔除。

    阈值取 0.25 而非 0.3 以更积极地去除期刊模板行。
    """
    lines = text.split("\n")
    if len(lines) < 10:
        return text

    line_counts = Counter(lines)
    threshold = max(3, int(len(lines) * 0.25))

    result = []
    for line in lines:
        stripped = line.strip().lower()
        if line_counts[line] >= threshold:
            continue
        if any(noise.lower() in stripped for noise in FOOTER_NOISE):
            continue
        result.append(line)
    return "\n".join(result)


def _inject_pseudo_headings(text: str) -> str:
    """逐行匹配章节模式，注入 # 前缀生成伪 Markdown。

    每行只匹配第一个命中的模式，不重复标记。
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        matched = False
        for pattern, replacement in HEADING_PATTERNS:
            m = re.match(pattern, stripped)
            if m:
                result.append(m.expand(replacement))
                matched = True
                break
        if not matched:
            result.append(line)
    return "\n".join(result)


def _detect_toc_page(text_lines: list[str]) -> bool:
    """检测是否为目录页：超过 10 个 '数字.数字' 模式 + 以数字结尾的行。"""
    pattern_count = 0
    total_lines = max(1, len(text_lines))
    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r'\d+\.\d+', stripped) and re.search(r'\d+$', stripped):
            pattern_count += 1
    return pattern_count >= 10 and pattern_count / total_lines >= 0.15


def _extract_title_by_font_size(pdf_path: str, total_pages: int) -> str:
    """从封面页（前3页）按字号提取论文标题。

    中文学位论文标题通常:
      - 在封面页上部
      - 字号最大（显著大于正文）
      - 跨1-2行
      - 不包含 "硕士学位论文""学号""指导教师" 等元数据词
    """
    _EXCLUDE_WORDS = [
        "硕士学位论文", "博士学位论文", "学士学位论文",
        "学号", "作者", "指导教师", "学院", "专业",
        "研究方向", "培养单位", "完成时间", "密级", "分类号",
        "学校代码", "图书分类", "UDC", "答辩",
        "申请学位", "论文答辩", "学位门类",
    ]
    doc = fitz.open(pdf_path)
    candidates: list[tuple[float, float, str]] = []

    for page_num in range(min(3, total_pages)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_h = page.rect.height

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # 取行中最大字号
                max_size = max(
                    (s.get("size", 0) for s in line["spans"]),
                    default=0,
                )
                # 取行顶部位置
                top = min(
                    (s["bbox"][1] for s in line["spans"]),
                    default=0,
                )
                text = "".join(s["text"] for s in line["spans"]).strip()

                # 过滤: 太短 / 含元数据词 / URL / 纯数字
                if len(text) < 5:
                    continue
                low = text.lower().replace(" ", "")
                if any(w.lower().replace(" ", "") in low for w in _EXCLUDE_WORDS):
                    continue
                if text.startswith("http") or text.startswith("www"):
                    continue
                # 仅当文本以英文为主（>50% ASCII字母）且字号<12时才跳过
                # 中英混合标题如"基于 XRD 的薄膜表征"不应被误杀
                ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
                if ascii_alpha > len(text) * 0.5 and max_size < 12:
                    continue
                rel_y = top / page_h if page_h > 0 else 1
                candidates.append((max_size, rel_y, text))

    doc.close()

    if not candidates:
        return ""

    # 字号大 + 位置靠上 = 标题（加权排序）
    scored = []
    for size, rel_y, text in candidates:
        pos_score = 1.0 - rel_y if rel_y < 0.5 else 0.2  # 靠上的优先
        len_score = min(len(text.replace(" ", "")) / 20, 1.0)
        score = size * 3.0 + pos_score * 5.0 + len_score * 2.0
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 合并相邻的大字号行作为标题（标题可能跨行）
    top_texts = [scored[0][1]]
    for _, text in scored[1:3]:
        # 如果字号相近且都是中文文本，可能是标题的续行
        if len(text.replace(" ", "")) >= 4 and \
           all('一' <= c <= '鿿' or c.isspace() for c in text if c.strip()):
            top_texts.append(text)
            if sum(len(t.replace(" ", "")) for t in top_texts) >= 15:
                break

    title = "".join(t.replace(" ", "") for t in top_texts)
    return title[:200]


def _is_blank_page(text: str) -> bool:
    """检测是否为空白页（<50 字符）。"""
    return len(text.strip()) < 50


def _is_cover_page(text: str, page_index: int, total_pages: int) -> bool:
    """检测是否为封面/声明页：前 3 页且无章节标题。"""
    if page_index >= 3:
        return False
    has_heading = any(
        re.match(pattern, line.strip())
        for line in text.split("\n")
        for pattern, _ in HEADING_PATTERNS
    )
    return not has_heading


def extract_text_pymupdf(pdf_path: str) -> tuple[str, dict]:
    """PyMuPDF 兜底提取：文本清洗 + 伪 MD 注入。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        (markdown_text, metadata_dict)
    """
    doc: fitz.Document = fitz.open(pdf_path)
    pages = len(doc)
    pdf_meta_raw = doc.metadata
    full_parts = []
    page_texts = []

    for i, page in enumerate(doc):
        text = page.get_text()
        page_texts.append(text)
        full_parts.append(text)

    doc.close()

    # 合并全文本
    full_text = "\n\n".join(full_parts)

    # 清洗页眉页脚
    cleaned = _clean_page_headers(full_text)

    # 移除目录页：从第一页起，跳过连续的目录/封面/空白页
    toc_end_idx = 0
    for i, pt in enumerate(page_texts):
        page_lines = pt.split("\n")
        if _detect_toc_page(page_lines) or _is_blank_page(pt) or _is_cover_page(pt, i, pages):
            continue
        else:
            toc_end_idx = i
            break

    if toc_end_idx > 0:
        full_text = "\n\n".join(full_parts[toc_end_idx:])
        cleaned = _clean_page_headers(full_text)
    else:
        cleaned = _clean_page_headers("\n\n".join(full_parts))

    # 注入伪 Markdown 标题
    markdown_text = _inject_pseudo_headings(cleaned)

    # 提取元数据
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "pages": pages,
        "file_path": pdf_path,
        "extraction_method": "pymupdf_fallback",
    }

    # 标题：PDF meta → 字号提取（前3页） → 伪MD标题 → 首行文本
    pdf_meta = pdf_meta_raw if pdf_meta_raw else {}
    if pdf_meta.get("title") and len(pdf_meta.get("title", "").strip()) >= 5:
        metadata["title"] = pdf_meta["title"].strip()[:200]
    else:
        # 用字号提取封面标题
        title_by_font = _extract_title_by_font_size(pdf_path, pages)
        if title_by_font:
            metadata["title"] = title_by_font[:200]
        else:
            _ABSTRACT_LIKE = {"abstract", "introduction", "method", "experiment",
                               "result", "discussion", "conclusion", "reference",
                               "摘要", "引言", "方法", "实验", "结果", "讨论",
                               "结论", "参考文献", "致谢", "目录", "附录"}
            for line in markdown_text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("## "):
                    candidate = stripped[3:].strip()
                    if candidate and candidate.lower() not in _ABSTRACT_LIKE and len(candidate) >= 5:
                        metadata["title"] = candidate[:200]
                        break

    if pdf_meta.get("author"):
        metadata["authors"] = [a.strip() for a in pdf_meta["author"].split(";")]

    # 摘要：从伪 MD 中找 ## Abstract / ## 摘要
    from .metadata import extract_abstract_from_md
    abstract = extract_abstract_from_md(markdown_text)
    if abstract:
        metadata["abstract"] = abstract
    else:
        # LLM 兜底：取全文前几千字符让 LLM 提取
        metadata["_full_text_for_llm_extraction"] = cleaned[:6000]

    return markdown_text, metadata
