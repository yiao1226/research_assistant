"""Markdown 感知分块 — 标题栈追踪 + CJK Token 估算 + 段落级重叠。

替換 ingestion.py 中原有的 _split_sections() + _split_long_text() + chunk_paper()。
"""
from __future__ import annotations

import re
import uuid
from typing import Optional


# ============================================================
# CJK-Aware Token 估算
# ============================================================

def _is_cjk(ch: str) -> bool:
    """检测字符是否为 CJK (中日韩) 统一汉字。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF    # CJK统一汉字
        or 0x3400 <= code <= 0x4DBF  # CJK扩展A
        or 0xF900 <= code <= 0xFAFF  # CJK兼容汉字
    )


def approx_token_len(text: str) -> int:
    """CJK 字符 = 1 token，非 CJK 按空白分词计数。

    BGE/DeepSeek 等 tokenizer 对中文几乎 1:1，英文约 0.7 token/word。
    此近似在混合中英文场景下误差 <15%，远优于 len(text) 字符数。
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    # 非CJK部分：统计英文单词数
    non_cjk = " ".join(ch for ch in text if not _is_cjk(ch))
    non_cjk_tokens = len([t for t in non_cjk.split() if t])
    return cjk + non_cjk_tokens


# ============================================================
# Markdown 段落分割（标题栈追踪）
# ============================================================

# 章节关键词（用于检测无 # 前缀的章节标题）
SECTION_KEYWORDS = [
    "Abstract", "Introduction", "Related Work", "Method",
    "Experiment", "Result", "Discussion", "Conclusion",
    "References", "Acknowledgments", "Appendix",
    "摘要", "引言", "方法", "实验", "结果", "讨论",
    "结论", "参考文献", "致谢", "目录", "附录",
    "总结与展望", "背景", "相关工作",
]

# 标题检测正则：H1-H3
HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)


def _clean_heading_text(text: str) -> str:
    """清理标题中的多余符号，保留关键词。"""
    text = re.sub(r'[●•○◉❖▶▷►♦■□▪▫]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_md_paragraphs(markdown_text: str) -> list[dict]:
    """根据 #/##/### 标题层级分割段落，保持语义完整性。

    维护 heading_stack，每遇到 # 行更新栈。
    每遇到空行 flush 当前累积的段落。
    每个段落携带完整 heading_path。

    Args:
        markdown_text: 包含 Markdown 标题的文本

    Returns:
        [{
            "content": "段落文本...",
            "heading_path": "第一章 绪论 > 1.1 X射线探测器简介",
            "heading_level": 2,
            "char_start": 0,
            "char_end": 523,
        }, ...]
    """
    if not markdown_text:
        return []

    lines = markdown_text.split("\n")
    paragraphs = []
    heading_stack: list[tuple[int, str]] = []  # [(level, heading_text), ...]
    current_lines: list[str] = []
    char_offset = 0
    current_start = 0

    # 第一遍：检测目录页和参考文献边界
    toc_end_pos = 0
    ref_start_pos = len(markdown_text)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.search(r'\d+\.\d+', stripped) and re.search(r'\d+$', stripped):
            if i < 30 and toc_end_pos < i:
                toc_start = markdown_text.find(line)
                if toc_start >= 0:
                    toc_end_pos = max(toc_end_pos, toc_start + 200)

    # 找 References/参考文献 位置
    for pattern in [r'^##\s+References?\b', r'^##\s+参考文献']:
        m = re.search(pattern, markdown_text, re.MULTILINE | re.IGNORECASE)
        if m:
            ref_start_pos = min(ref_start_pos, m.start())
            break

    def _flush_paragraph():
        nonlocal current_start, current_lines
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text and len(text) > 10:
                heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else ""
                paragraphs.append({
                    "content": text,
                    "heading_path": heading_path,
                    "heading_level": heading_stack[-1][0] if heading_stack else 0,
                    "char_start": current_start,
                    "char_end": char_offset,
                })
            current_lines = []
            current_start = char_offset

    for line in lines:
        stripped = line.strip()
        char_offset += len(line) + 1  # +1 for \n

        # 跳过目录区域
        if char_offset < toc_end_pos:
            continue

        # 检测 # 标题行
        m = HEADING_RE.match(stripped)
        if m:
            _flush_paragraph()
            level = len(m.group(1))
            heading_text = _clean_heading_text(m.group(2))
            # 更新标题栈：移除同级或更深层的标题
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            continue

        # 检测无 # 前缀的章节关键词
        if not stripped.startswith("#"):
            kw_matched = False
            clean = _clean_heading_text(stripped)
            for kw in SECTION_KEYWORDS:
                if clean.lower() == kw.lower():
                    _flush_paragraph()
                    while heading_stack and heading_stack[-1][0] >= 2:
                        heading_stack.pop()
                    heading_stack.append((2, clean))
                    kw_matched = True
                    break
            if kw_matched:
                continue

        # 空行 → 段落边界
        if not stripped:
            _flush_paragraph()
            continue

        # 跳过纯数字行（页码、编号），但保留日期和较长数字串
        if re.match(r'^[\d\s\.\-]+$', stripped) and len(stripped) <= 12:
            continue
        if stripped.startswith("![") or stripped.startswith("http"):
            continue

        current_lines.append(stripped)

    # flush 最后一段
    _flush_paragraph()

    # 标记参考文献区域
    for p in paragraphs:
        if p["char_start"] >= ref_start_pos:
            if not p["heading_path"]:
                p["heading_path"] = "References"
            p["_is_reference"] = True

    return paragraphs


# ============================================================
# 段落级重叠分块
# ============================================================

def chunk_paragraphs(
    paragraphs: list[dict],
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    reference_max_tokens: int = 1024,
) -> list[dict]:
    """基于 Token 数量的段落级智能分块，含重叠策略。

    当前 chunk 填满 chunk_tokens 后，从尾部取 overlap_tokens 个段落回溯，
    作为下一个 chunk 的开头，保证语义不跨边界断裂。

    Args:
        paragraphs: split_md_paragraphs 的输出
        chunk_tokens: 目标 token 数（默认 512）
        overlap_tokens: 重叠 token 数（默认 64）
        reference_max_tokens: 参考文献区域放宽上限

    Returns:
        [{
            "content": "合并后的段落文本",
            "heading_path": "第二章 > 2.3 电学性质",
            "char_start": 15200,
            "char_end": 16800,
            "is_reference": False,
        }, ...]
    """
    if not paragraphs:
        return []

    chunks = []
    i = 0

    while i < len(paragraphs):
        current_paras = []
        current_tokens = 0
        current_start = paragraphs[i]["char_start"]
        is_ref = paragraphs[i].get("_is_reference", False)
        max_tok = reference_max_tokens if is_ref else chunk_tokens

        # 向前累积段落直到达到目标 token 数
        while i < len(paragraphs) and current_tokens < max_tok:
            p = paragraphs[i]
            p_tokens = approx_token_len(p["content"])

            # 单个超大段落：按 token 强制切割
            if not current_paras and p_tokens > max_tok * 2 and not p.get("_is_reference"):
                sub_chunks = _split_long_paragraph(p, chunk_tokens)
                for sc in sub_chunks:
                    chunks.append({
                        "content": sc["content"],
                        "heading_path": p["heading_path"],
                        "char_start": sc.get("char_start", current_start),
                        "char_end": sc.get("char_end", current_start),
                        "is_reference": False,
                    })
                i += 1
                continue

            current_paras.append(p)
            current_tokens += p_tokens
            i += 1

        if not current_paras:
            continue

        # 回溯 overlap_tokens 个段落
        backtrack_start = max(0, len(current_paras) - max(1, overlap_tokens // 50))
        # 向前回溯 i 指针
        if backtrack_start < len(current_paras) - 1:
            overlap_count = len(current_paras) - backtrack_start
            i = i - overlap_count

        # 合并当前 chunk
        content = "\n\n".join(p["content"] for p in current_paras)
        # 取首个有 heading_path 的段落作为 chunk 标题
        first_heading = ""
        last_heading = ""
        for p in current_paras:
            hp = p.get("heading_path", "")
            if hp:
                if not first_heading:
                    first_heading = hp
                last_heading = hp
        # 如果首尾不同，显示范围；否则只显示首个
        if first_heading and last_heading and first_heading != last_heading:
            heading_path = f"{first_heading} … {last_heading}"
        else:
            heading_path = first_heading

        chunks.append({
            "content": content,
            "heading_path": heading_path,
            "char_start": current_paras[0]["char_start"],
            "char_end": current_paras[-1]["char_end"],
            "is_reference": is_ref,
        })

    return chunks


def _split_long_paragraph(paragraph: dict, chunk_tokens: int) -> list[dict]:
    """将单个超长段落按 token 数量切割（回退策略）。"""
    text = paragraph["content"]
    if not text:
        return []

    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    chunks = []
    current = []
    current_tokens = 0
    start = paragraph.get("char_start", 0)

    for sent in sentences:
        st = approx_token_len(sent)
        if current_tokens + st > chunk_tokens and current:
            content = " ".join(current)
            chunks.append({
                "content": content,
                "char_start": start,
                "char_end": start + len(content),
            })
            start += len(content)
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += st

    if current:
        content = " ".join(current)
        chunks.append({
            "content": content,
            "char_start": start,
            "char_end": start + len(content),
        })

    return chunks


# ============================================================
# 统一入口：替換 ingestion.py 的 chunk_paper
# ============================================================

def chunk_paper(
    markdown_text: str,
    paper_id: int,
    annotation: dict | None = None,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[dict]:
    """论文分块统一入口 — 替換 ingestion.py 中的 chunk_paper()。

    内部链路:
      split_md_paragraphs() → chunk_paragraphs()
      → 每段 uuid5 生成确定性 ID
      → payload 带 paper_id / heading_path / chunk_index / contribution_type

    Args:
        markdown_text: Markdown 格式的论文全文
        paper_id: 论文 ID（用于生成确定性 UUID）
        annotation: 论文标注字典（可选）
        chunk_tokens: 每 chunk 目标 token 数
        overlap_tokens: 重叠 token 数

    Returns:
        [{"id": uuid5, "text": "...", "payload": {...}}, ...]
    """
    if not markdown_text:
        return []

    ann = annotation or {}
    paragraphs = split_md_paragraphs(markdown_text)
    if not paragraphs:
        return []

    chunks = chunk_paragraphs(
        paragraphs,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    result = []
    namespace = uuid.NAMESPACE_DNS
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid5(namespace, f"paper_{paper_id}_chunk_{i}"))
        result.append({
            "id": chunk_id,
            "text": chunk["content"],
            "payload": {
                "paper_id": paper_id,
                "heading_path": chunk.get("heading_path", ""),
                "chunk_index": i,
                "chunk_count": len(chunks),
                "contribution_type": ann.get("contribution_type", ""),
                "core_claim": ann.get("core_claim", ""),
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "is_reference": chunk.get("is_reference", False),
            },
        })
    return result


# ============================================================
# 句子窗口检索 — LlamaIndex Sentence Window Retrieval
# ============================================================
# 索引粒度: 单个句子（高精度检索）
# 上下文窗口: 前后各 N 句（宽上下文生成）
# 检索时 Qdrant 返回句子节点，后处理用 window_text 替换

def _split_sentences(text: str) -> list[str]:
    """中英文混合分句。

    仅在句末标点处切分。不再以换行符为边界——PDF 提取文本
    有大量排版换行，按 \\n 切分会把一句切成多个碎片。
    """
    import re
    # 保护小数点、缩写等
    text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
    text = re.sub(r'(Mr|Dr|Prof|vs|Fig|Eq|et al)\.', r'\1<DOT>', text, flags=re.IGNORECASE)

    # 分句：仅按句末标点切分。换行符替换为空格避免粘连
    text = text.replace('\n', ' ')
    sentences = re.split(
        r'(?<=[。！？.!?])\s+',
        text,
    )

    # 还原并过滤
    result = []
    for s in sentences:
        s = s.replace('<DOT>', '.').strip()
        if s and len(s) >= 3:
            result.append(s)
    return result


def chunk_paper_sentence_window(
    markdown_text: str,
    paper_id: int,
    annotation: dict | None = None,
    window_size: int = 3,
) -> list[dict]:
    """句子窗口分块 — 替換 chunk_paper。

    索引单位: 单个句子（嵌入到 Qdrant）
    上下文窗口: 前后各 window_size 句（存储在 payload.window_text）

    检索流程:
      1. Qdrant 返回匹配的句子（高精度）
      2. post_process_sentence_window() 用 window_text 替换句子文本
      3. LLM 收到带上下文的宽窗口文本

    Args:
        markdown_text: Markdown 格式的论文全文
        paper_id: 论文 ID
        annotation: 标注字典
        window_size: 前后句子数（默认3，窗口=7句）

    Returns:
        Qdrant-ready points: [{"id": uuid5, "text": "单句",
          "payload": {"window_text": "前后各N句的上下文", ...}}, ...]
    """
    if not markdown_text:
        return []

    ann = annotation or {}

    # Step 1: 段落分割（保留 heading_path）
    paragraphs = split_md_paragraphs(markdown_text)
    if not paragraphs:
        return []

    # Step 2: 每个段落内分句，带 heading_path
    all_sentences = []
    for p in paragraphs:
        hp = p.get("heading_path", "")
        sents = _split_sentences(p["content"])
        for s in sents:
            all_sentences.append({
                "text": s,
                "heading_path": hp,
            })

    if not all_sentences:
        return []

    # Step 3: 构建窗口节点
    total = len(all_sentences)
    namespace = uuid.NAMESPACE_DNS
    result = []

    for i, sent in enumerate(all_sentences):
        # 计算窗口边界
        start = max(0, i - window_size)
        end = min(total, i + window_size + 1)

        # 构建窗口文本（带章节标记）
        window_parts = []
        for j in range(start, end):
            hp_j = all_sentences[j].get("heading_path", "")
            prefix = f"[{hp_j}] " if hp_j and hp_j != sent.get("heading_path", "") else ""
            marker = " ▶ " if j == i else "   "
            window_parts.append(f"{marker}{prefix}{all_sentences[j]['text']}")

        window_text = "\n".join(window_parts)

        chunk_id = str(uuid.uuid5(namespace, f"paper_{paper_id}_sent_{i}"))

        result.append({
            "id": chunk_id,
            "text": sent["text"],          # ← 向量检索目标（单句）
            "payload": {
                "paper_id": paper_id,
                "heading_path": sent.get("heading_path", ""),
                "sentence_index": i,
                "sentence_count": total,
                "window_text": window_text,  # ← LLM 上下文（宽窗口）
                "window_start": start,
                "window_end": end,
                "contribution_type": ann.get("contribution_type", ""),
                "core_claim": ann.get("core_claim", ""),
                "is_reference": False,
                "chunk_strategy": "sentence_window",
            },
        })

    return result


def post_process_sentence_window(results: list[dict]) -> list[dict]:
    """后处理: 用 window_text 替换检索到的句子文本。

    兼容两种结果格式:
      - 展平格式: {text, window_text, heading_path, ...} (dense_search 直接返回)
      - 嵌套格式: {payload: {text, window_text, ...}} (部分调用方)

    Args:
        results: Qdrant 搜索结果列表

    Returns:
        替换后的结果列表
    """
    for r in results:
        # 兼容展平 payload 和嵌套 payload
        if "window_text" in r:
            window = r["window_text"]
            if window:
                r["_original_text"] = r.get("text", "")
                r["text"] = window
                r["_is_window"] = True
        elif "payload" in r:
            payload = r.get("payload", {})
            window = payload.get("window_text", "")
            if window:
                payload["_original_text"] = payload.get("text", "")
                payload["text"] = window
                payload["_is_window"] = True
    return results
