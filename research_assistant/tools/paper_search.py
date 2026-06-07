"""真实论文检索 — ArXiv + Semantic Scholar + Web of Science + 个性化排序。

特性:
  - Query Understanding: 结合用户研究画像，LLM 改写关键词
  - 多因素排序: 语义匹配 + 关键词 + 热门度 + 时效
  - Top-5 返回 + 排名理由
  - 用户手动选择入库（非自动）
  - 入库时调用 IngestionPipeline 生成厚标注
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import feedparser
import requests
from langchain_core.tools import tool

# === API 端点 ===
ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
WOS_API_URL = "https://api.clarivate.com/apis/wos-starter/v1"

_HTTP_HEADERS = {
    "User-Agent": "ResearchAssistant/2.0 (Academic Research Tool; mailto:research@example.com)"
}

# === 速率限制状态（线程安全） ===
_rate_lock = threading.Lock()


class _RateLimiter:
    """线程安全的速率限制器。"""

    def __init__(self, min_interval: float = 1.0, cooldown: float = 60.0):
        self._last_request = 0.0
        self._min_interval = min_interval
        self._429_until = 0.0
        self._429_cooldown = cooldown

    def wait(self, label: str = ""):
        """等待直到可以发送下一个请求。"""
        with _rate_lock:
            now = time.time()
            if now < self._429_until:
                wait = self._429_until - now + 2
                if label:
                    print(f"  [{label}] 处于限流冷却期，等待 {wait:.0f} 秒...", file=sys.stderr, flush=True)
                time.sleep(wait)
                now = time.time()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed + 0.5)
                now = time.time()
            self._last_request = now

    def mark_429(self):
        """标记收到 429 响应。"""
        with _rate_lock:
            self._429_until = time.time() + self._429_cooldown


_arxiv_limiter = _RateLimiter(min_interval=6.0, cooldown=60.0)
_s2_limiter = _RateLimiter(min_interval=1.5, cooldown=30.0)


def _http_get(url, params, headers=None, timeout=30, max_retries=3,
              is_arxiv=False, is_s2=False) -> dict:
    merged_headers = {**_HTTP_HEADERS, **(headers or {})}
    last_error = "未知错误"

    limiter = None
    label = ""
    if is_arxiv:
        limiter = _arxiv_limiter
        label = "ArXiv"
    elif is_s2:
        limiter = _s2_limiter
        label = "Semantic Scholar"

    for attempt in range(max_retries):
        if limiter:
            limiter.wait(label=label)

        try:
            resp = requests.get(url, params=params, headers=merged_headers, timeout=timeout)

            if resp.status_code == 429:
                last_error = "HTTP 429 (请求过于频繁)"
                if limiter:
                    limiter.mark_429()
                time.sleep(min(30 * (2 ** attempt), 120))
                continue

            if resp.status_code == 403:
                last_error = "HTTP 403 (访问被拒绝)"
                time.sleep(5 * (2 ** attempt))
                continue

            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code} (服务器错误)"
                time.sleep(5 * (2 ** attempt))
                continue

            if resp.status_code == 200:
                return {"ok": True, "response": resp}

            last_error = f"HTTP {resp.status_code}"
            return {"ok": False, "error": last_error}

        except requests.Timeout:
            last_error = "请求超时"
            time.sleep(5 * (2 ** attempt))
        except requests.ConnectionError:
            last_error = "网络连接失败"
            time.sleep(5 * (2 ** attempt))

    return {"ok": False, "error": f"{last_error}（已重试 {max_retries} 次）"}


# ============================================================
# 论文下载
# ============================================================

DOWNLOAD_DIR = Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


@tool
def download_paper(arxiv_id: str) -> str:
    """下载 ArXiv 论文 PDF 到本地 ./downloads 目录。

    参数:
        arxiv_id: ArXiv 论文 ID，例如 '2301.12345'

    返回:
        下载状态和文件路径。
    """
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return json.dumps({"status": "error", "message": "ArXiv ID 不能为空。"}, ensure_ascii=False)

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filepath = DOWNLOAD_DIR / f"{arxiv_id}.pdf"

    if filepath.exists():
        return json.dumps({
            "status": "ok",
            "message": f"论文已存在: {filepath}",
            "file_path": str(filepath),
            "arxiv_id": arxiv_id,
        }, ensure_ascii=False)

    result = _http_get(pdf_url, {}, timeout=60, max_retries=2, is_arxiv=True)
    if not result["ok"]:
        return json.dumps({"status": "error", "message": f"下载失败: {result['error']}"}, ensure_ascii=False)

    try:
        content = result["response"].content
        # 大小限制：拒绝 >100MB 的 PDF
        max_bytes = 100 * 1024 * 1024
        if len(content) > max_bytes:
            return json.dumps({
                "status": "error",
                "message": f"文件过大 ({len(content)/1024/1024:.1f} MB)，超过 100 MB 限制",
            }, ensure_ascii=False)
        filepath.write_bytes(content)
        size_kb = len(content) / 1024
        return json.dumps({
            "status": "ok",
            "message": f"下载成功: {arxiv_id}.pdf ({size_kb:.1f} KB)",
            "file_path": str(filepath),
            "arxiv_id": arxiv_id,
            "size_kb": round(size_kb, 1),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"保存文件失败: {e}"}, ensure_ascii=False)


@tool
def fetch_paper_full_text(arxiv_id: str) -> str:
    """通过 ArXiv ID 获取论文完整元数据（标题、作者、摘要等）。

    参数:
        arxiv_id: ArXiv 论文 ID，例 '2301.12345'
    """
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return json.dumps({"status": "error", "message": "ArXiv ID 不能为空。"}, ensure_ascii=False)

    params = {"search_query": f"id:{arxiv_id}", "start": 0, "max_results": 1}
    result = _http_get(ARXIV_API_URL, params, timeout=30, max_retries=2, is_arxiv=True)
    if not result["ok"]:
        return json.dumps({"status": "error", "message": result["error"]}, ensure_ascii=False)

    feed = feedparser.parse(result["response"].text)
    if not feed.entries:
        return json.dumps({
            "status": "error",
            "message": f"未找到 ArXiv ID 为 {arxiv_id} 的论文。"
        }, ensure_ascii=False)

    entry = feed.entries[0]
    return json.dumps({
        "status": "ok",
        "arxiv_id": arxiv_id,
        "title": entry.title.strip(),
        "authors": [a.name for a in entry.authors] if hasattr(entry, 'authors') else [],
        "abstract": entry.summary.strip() if hasattr(entry, 'summary') else "",
        "published": entry.published if hasattr(entry, 'published') else "",
        "doi": getattr(entry, 'arxiv_doi', ''),
        "categories": [t.term for t in entry.tags] if hasattr(entry, 'tags') else [],
    }, ensure_ascii=False, indent=2)


# ============================================================
# ArXiv 搜索
# ============================================================

@tool
def search_arxiv(query: str, max_results: int = 15, days_back: int = 730,
                 sort_by: str = "relevance") -> str:
    """在 ArXiv 上搜索真实学术论文。

    参数:
        query: 搜索关键词，支持 AND/OR 布尔运算
        max_results: 最大返回数量（默认15，最多30）
        days_back: 仅返回最近 N 天的论文（0=不限）
        sort_by: 'relevance' 或 'submittedDate'

    返回:
        JSON，status="ok" 时 papers 数组包含真实论文。
    """
    query = query.strip()
    if not query:
        return json.dumps({"status": "error", "message": "搜索关键词不能为空。"}, ensure_ascii=False)

    max_results = min(max(1, int(max_results)), 30)

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    result = _http_get(ARXIV_API_URL, params, timeout=30, max_retries=3, is_arxiv=True)
    if not result["ok"]:
        return json.dumps({
            "status": "error",
            "message": f"ArXiv 无法访问: {result['error']}。"
        }, ensure_ascii=False)

    feed = feedparser.parse(result["response"].text)
    if not feed.entries:
        return json.dumps({"status": "ok", "total_results": 0, "papers": []}, ensure_ascii=False)

    cutoff = None
    if int(days_back) > 0:
        cutoff = datetime.now() - timedelta(days=int(days_back))

    papers = []
    for entry in feed.entries:
        pub_date = None
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            pub_date = datetime(*entry.published_parsed[:6])
        if cutoff and pub_date and pub_date < cutoff:
            continue

        arxiv_id = (
            entry.id.split("/abs/")[-1] if "/abs/" in entry.id
            else entry.id.split("/")[-1]
        )
        papers.append({
            "arxiv_id": arxiv_id,
            "title": entry.title.strip().replace("\n", " "),
            "authors": [a.name for a in entry.authors] if hasattr(entry, 'authors') else [],
            "abstract": entry.summary.strip().replace("\n", " ") if hasattr(entry, 'summary') else "",
            "published": entry.published if hasattr(entry, 'published') else "",
            "year": pub_date.year if pub_date else None,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "doi": getattr(entry, 'arxiv_doi', ''),
            "categories": [t.term for t in entry.tags] if hasattr(entry, 'tags') else [],
            "source": "arxiv",
        })

    return json.dumps({
        "status": "ok",
        "total_results": len(papers),
        "papers": papers,
    }, ensure_ascii=False, indent=2)


# ============================================================
# Semantic Scholar 搜索
# ============================================================

@tool
def search_semantic_scholar(query: str, max_results: int = 10, fields: str = "") -> str:
    """在 Semantic Scholar 上搜索真实学术论文（含引用数据）。

    参数:
        query: 搜索关键词
        max_results: 最大返回数量（默认10，最多20）

    返回:
        JSON，含论文列表及真实引用统计。
    """
    query = query.strip()
    if not query:
        return json.dumps({"status": "error", "message": "搜索关键词不能为空。"}, ensure_ascii=False)

    if not fields:
        fields = "title,authors,abstract,year,citationCount,influentialCitationCount,externalIds,url,publicationVenue"

    max_results = min(max(1, int(max_results)), 20)
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {"query": query, "limit": max_results, "fields": fields}

    result = _http_get(url, params, timeout=30, max_retries=3, is_s2=True)
    if not result["ok"]:
        return json.dumps({"status": "error", "message": f"Semantic Scholar 无法访问: {result['error']}."}, ensure_ascii=False)

    data = result["response"].json()
    papers = []
    for item in data.get("data", []):
        papers.append({
            "title": item.get("title", ""),
            "authors": [a.get("name", "") for a in item.get("authors", [])],
            "abstract": item.get("abstract", ""),
            "year": item.get("year"),
            "citation_count": item.get("citationCount", 0),
            "influential_citations": item.get("influentialCitationCount", 0),
            "doi": item.get("externalIds", {}).get("DOI", ""),
            "arxiv_id": item.get("externalIds", {}).get("ArXiv", ""),
            "url": item.get("url", ""),
            "venue": item.get("publicationVenue", {}).get("name", "") if item.get("publicationVenue") else "",
            "source": "semantic_scholar",
        })

    return json.dumps({
        "status": "ok",
        "total_results": len(papers),
        "papers": papers,
    }, ensure_ascii=False, indent=2)


# ============================================================
# Web of Science 搜索
# ============================================================

@tool
def search_web_of_science(query: str, max_results: int = 10) -> str:
    """在 Web of Science 上搜索真实学术论文（需要 WOS_API_KEY 环境变量）。

    参数:
        query: 搜索关键词
        max_results: 最大返回数量（默认10，最多25）
    """
    api_key = os.getenv("WOS_API_KEY", "")
    if not api_key:
        return json.dumps({
            "status": "error",
            "message": "Web of Science 需要 WOS_API_KEY 环境变量。"
        }, ensure_ascii=False)

    query = query.strip()
    if not query:
        return json.dumps({"status": "error", "message": "搜索关键词不能为空。"}, ensure_ascii=False)

    max_results = min(max(1, int(max_results)), 25)
    url = f"{WOS_API_URL}/documents"
    params = {"db": "WOS", "q": f"TS=({query})", "limit": max_results, "page": 1}
    headers = {"X-ApiKey": api_key}

    result = _http_get(url, params, headers=headers, timeout=30, max_retries=2)
    if not result["ok"]:
        return json.dumps({"status": "error", "message": f"Web of Science 无法访问: {result['error']}."}, ensure_ascii=False)

    data = result["response"].json()
    papers = []
    for hit in data.get("hits", []):
        papers.append({
            "title": hit.get("title", ""),
            "authors": [a.get("displayName", "") for a in hit.get("authors", {}).get("authors", [])],
            "abstract": hit.get("abstract", ""),
            "doi": hit.get("doi", ""),
            "published": hit.get("source", {}).get("publishedDate", ""),
            "venue": hit.get("source", {}).get("sourceTitle", ""),
            "citation_count": hit.get("citationCount", 0),
            "source": "web_of_science",
        })

    return json.dumps({
        "status": "ok",
        "total_results": len(papers),
        "papers": papers,
    }, ensure_ascii=False, indent=2)
