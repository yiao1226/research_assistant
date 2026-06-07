"""Per-user storage layer — SQLite for structured data + JSON for backups.

SQLite tables (per user library.db):
  - papers:       论文元数据 + 标注
  - progress:     用户研究进展
  - plans:        研究计划
  - search_history: 搜索历史
  - session_log:  操作日志

JSON backups:
  - backups/ 目录下按日期存放全量快照
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class PerUserStorage:
    """每个用户独立的 SQLite + JSON 存储。"""

    def __init__(self, user_dir: str | Path):
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.user_dir / "library.db")
        self.backup_dir = self.user_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self._init_db()

    # === SQLite 连接 ===

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # === 表初始化 ===

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT,
                    doi TEXT,
                    title TEXT NOT NULL,
                    authors TEXT,           -- JSON array
                    abstract TEXT,
                    published TEXT,
                    year INTEGER,
                    pdf_url TEXT,
                    venue TEXT,
                    citation_count INTEGER DEFAULT 0,
                    influential_citations INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'unknown',  -- arxiv/s2/wos/user_upload
                    annotation TEXT,         -- JSON: PaperAnnotation
                    annotation_quality TEXT DEFAULT 'none', -- full_text/abstract_only/none
                    file_path TEXT,          -- 本地 PDF 路径（如有）
                    ingested_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(arxiv_id, doi)
                );
                CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
                CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
                CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
                CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);

                -- FTS5 全文索引（替换 LIKE '%kw%' 全表扫描）
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    title, abstract, annotation,
                    content='papers',
                    content_rowid='id',
                );
                -- 触发器：papers 变更时自动同步 FTS5
                CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                    INSERT INTO papers_fts(rowid, title, abstract, annotation)
                    VALUES (new.id, new.title, new.abstract, new.annotation);
                END;
                CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, annotation)
                    VALUES ('delete', old.id, old.title, old.abstract, old.annotation);
                END;
                CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, annotation)
                    VALUES ('delete', old.id, old.title, old.abstract, old.annotation);
                    INSERT INTO papers_fts(rowid, title, abstract, annotation)
                    VALUES (new.id, new.title, new.abstract, new.annotation);
                END;
                -- 存量数据填充 FTS5（幂等：已存在的 rowid 会跳过）
                INSERT OR IGNORE INTO papers_fts(rowid, title, abstract, annotation)
                    SELECT id, title, abstract, annotation FROM papers;

                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    topic TEXT NOT NULL,
                    entry_type TEXT DEFAULT 'other',
                    title TEXT NOT NULL,
                    content TEXT,
                    results TEXT,
                    insights TEXT,
                    next_actions TEXT,
                    related_papers TEXT,     -- JSON: [arxiv_id, ...]
                    tags TEXT,               -- JSON: [tag1, tag2, ...]
                    metrics TEXT             -- JSON: {key: value, ...}
                );
                CREATE INDEX IF NOT EXISTS idx_progress_topic ON progress(topic);
                CREATE INDEX IF NOT EXISTS idx_progress_timestamp ON progress(timestamp);

                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    phases TEXT,             -- JSON: [Phase, ...]
                    current_phase INTEGER DEFAULT 0,
                    overall_progress REAL DEFAULT 0.0,
                    active BOOLEAN DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_plans_topic ON plans(topic);

                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    query TEXT NOT NULL,
                    sources TEXT,            -- JSON: ["arxiv", "s2"]
                    num_results INTEGER,
                    num_selected INTEGER,
                    duration_sec REAL
                );

                CREATE TABLE IF NOT EXISTS session_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    op_type TEXT NOT NULL,   -- search/upload/review/progress/plan/recall/system
                    summary TEXT,
                    details TEXT,            -- JSON
                    duration_sec REAL,
                    success BOOLEAN DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_session_type ON session_log(op_type);
                CREATE INDEX IF NOT EXISTS idx_session_ts ON session_log(timestamp);
            """)

    # === Paper CRUD ===

    def add_paper(self, paper: dict) -> int:
        """添加论文，去重检测基于 arxiv_id + doi。返回 paper_id。"""
        with self._conn() as c:
            # 去重
            aid = paper.get("arxiv_id") or None
            doi = paper.get("doi") or None
            if aid or doi:
                existing = c.execute(
                    "SELECT id FROM papers WHERE (arxiv_id=? AND arxiv_id IS NOT NULL) OR (doi=? AND doi IS NOT NULL)",
                    (aid, doi),
                ).fetchone()
                if existing:
                    # 更新 annotation 如果有新的
                    if paper.get("annotation"):
                        c.execute(
                            "UPDATE papers SET annotation=?, annotation_quality=?, updated_at=datetime('now') WHERE id=?",
                            (json.dumps(paper["annotation"], ensure_ascii=False),
                             paper.get("annotation_quality", "abstract_only"),
                             existing["id"]),
                        )
                    return existing["id"]

            c.execute(
                """INSERT INTO papers (arxiv_id, doi, title, authors, abstract, published, year,
                   pdf_url, venue, citation_count, influential_citations, source,
                   annotation, annotation_quality, file_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    aid,
                    doi,
                    paper.get("title", ""),
                    json.dumps(paper.get("authors", []), ensure_ascii=False),
                    paper.get("abstract", ""),
                    paper.get("published", ""),
                    paper.get("year"),
                    paper.get("pdf_url", ""),
                    paper.get("venue", ""),
                    paper.get("citation_count", 0),
                    paper.get("influential_citations", 0),
                    paper.get("source", "unknown"),
                    json.dumps(paper.get("annotation"), ensure_ascii=False) if paper.get("annotation") else None,
                    paper.get("annotation_quality", "none"),
                    paper.get("file_path", ""),
                ),
            )
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_paper_by_id(self, paper_id: int) -> dict | None:
        """通过 ID 获取论文。"""
        with self._conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def get_papers_by_ids(self, paper_ids: list[int]) -> list[dict]:
        """批量获取论文。"""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM papers WHERE id IN ({','.join('?'*len(paper_ids))})",
                paper_ids,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search_papers_local(self, keywords: list[str], limit: int = 20) -> list[dict]:
        """SQLite 全文关键词搜索，优先使用 FTS5 索引。"""
        with self._conn() as c:
            # 尝试 FTS5（存量数据可能还未索引，回退 LIKE）
            try:
                fts_query = " OR ".join(kw.strip() for kw in keywords if kw.strip())
                if fts_query:
                    # 同时查询 FTS5 获取 rowid，JOIN papers 获取完整数据
                    rows = c.execute(
                        """SELECT p.* FROM papers p
                           INNER JOIN papers_fts f ON p.id = f.rowid
                           WHERE papers_fts MATCH ?
                           ORDER BY p.year DESC, p.citation_count DESC
                           LIMIT ?""",
                        (fts_query, limit),
                    ).fetchall()
                    if rows:
                        return [self._row_to_dict(r) for r in rows]
            except Exception:
                pass  # FTS5 不可用或查询语法错误，回退 LIKE

            # 回退：LIKE 全表扫描
            conditions = " OR ".join(
                ["(title LIKE ? OR abstract LIKE ? OR annotation LIKE ?)" for _ in keywords]
            )
            params = []
            for kw in keywords:
                params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])
            rows = c.execute(
                f"SELECT * FROM papers WHERE {conditions} ORDER BY year DESC, citation_count DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_papers(self) -> list[dict]:
        """获取所有论文。"""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM papers ORDER BY year DESC, citation_count DESC").fetchall()
            return [self._row_to_dict(r) for r in rows]

    def count_papers(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def get_paper(self, paper_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM papers WHERE id=?", (paper_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def delete_paper(self, paper_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM papers WHERE id=?", (paper_id,))
            return cur.rowcount > 0

    # === Progress CRUD ===

    def add_progress(self, entry: dict) -> int:
        with self._conn() as c:
            c.execute(
                """INSERT INTO progress (topic, entry_type, title, content, results, insights,
                   next_actions, related_papers, tags, metrics)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.get("topic", ""),
                    entry.get("entry_type", "other"),
                    entry.get("title", ""),
                    entry.get("content", ""),
                    entry.get("results"),
                    entry.get("insights"),
                    entry.get("next_actions"),
                    json.dumps(entry.get("related_papers", []), ensure_ascii=False),
                    json.dumps(entry.get("tags", []), ensure_ascii=False),
                    json.dumps(entry.get("metrics", {}), ensure_ascii=False),
                ),
            )
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_progress_by_topic(self, topic: str, limit: int = 30) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM progress WHERE topic=? ORDER BY timestamp DESC LIMIT ?",
                (topic, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_progress(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM progress ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # === Plan CRUD ===

    def save_plan(self, topic: str, phases: list[dict], current_phase: int = 0,
                  overall_progress: float = 0.0) -> int:
        with self._conn() as c:
            # 同一 topic 只保留一份活跃计划
            c.execute("UPDATE plans SET active=0 WHERE topic=? AND active=1", (topic,))
            c.execute(
                "INSERT INTO plans (topic, phases, current_phase, overall_progress) VALUES (?,?,?,?)",
                (topic, json.dumps(phases, ensure_ascii=False), current_phase, overall_progress),
            )
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_active_plan(self, topic: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM plans WHERE topic=? AND active=1 ORDER BY updated_at DESC LIMIT 1",
                (topic,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def update_plan_progress(self, plan_id: int, phases: list[dict],
                             current_phase: int, overall_progress: float):
        with self._conn() as c:
            c.execute(
                "UPDATE plans SET phases=?, current_phase=?, overall_progress=?, updated_at=datetime('now') WHERE id=?",
                (json.dumps(phases, ensure_ascii=False), current_phase, overall_progress, plan_id),
            )

    # === Search History ===

    def log_search(self, query: str, sources: list[str], num_results: int,
                   num_selected: int, duration_sec: float):
        with self._conn() as c:
            c.execute(
                "INSERT INTO search_history (query, sources, num_results, num_selected, duration_sec) VALUES (?,?,?,?,?)",
                (query, json.dumps(sources), num_results, num_selected, duration_sec),
            )

    def get_search_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM search_history ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # === Session Log ===

    def log_operation(self, op_type: str, summary: str, details: dict | None = None,
                      duration_sec: float = 0.0, success: bool = True):
        with self._conn() as c:
            c.execute(
                "INSERT INTO session_log (op_type, summary, details, duration_sec, success) VALUES (?,?,?,?,?)",
                (op_type, summary, json.dumps(details, ensure_ascii=False) if details else None,
                 duration_sec, 1 if success else 0),
            )

    # === Backup ===

    def export_all_json(self) -> dict:
        """导出全量数据为 JSON 字典。"""
        data = {}
        with self._conn() as c:
            for table in ["papers", "progress", "plans", "search_history", "session_log"]:
                rows = c.execute(f"SELECT * FROM {table}").fetchall()
                data[table] = [self._row_to_dict(r) for r in rows]
        return data

    def write_backup(self):
        """写入 JSON 备份文件。"""
        today = datetime.now().strftime("%Y%m%d")
        data = self.export_all_json()
        # 分文件备份
        for table, rows in data.items():
            if rows:
                fpath = self.backup_dir / f"{table}_{today}.json"
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
        # 清理 7 天前的备份
        self._cleanup_backups(7)

    def _cleanup_backups(self, days: int = 7):
        cutoff = time.time() - days * 86400
        for f in self.backup_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()

    # === Helpers ===

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        if row is None:
            return None
        d = dict(row)
        # 反序列化 JSON 字段
        for field in ["authors", "annotation", "phases", "related_papers", "tags",
                      "metrics", "sources", "details"]:
            if field in d and d[field] and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        return d
