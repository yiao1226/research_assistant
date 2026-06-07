"""多层级备份与操作日志管理器。

备份策略（4层）:
  第1层: SQLite WAL 模式 — 事务级持久，断电不丢数据
  第2层: Qdrant 磁盘持久化 — 可从 SQLite 重建
  第3层: JSON 快照 — 每日自动 + 手动触发，保留7天
  第4层: 操作日志 — 每次操作记录，可审计可回放

日志文件:
  data/users/{name}/logs/
    search_{YYYYMM}.log
    upload_{YYYYMM}.log
    review_{YYYYMM}.log
    system_{YYYYMM}.log
    backup_{YYYYMM}.log
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .storage import PerUserStorage


class BackupManager:
    """备份与日志管理。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.user_dir = storage.user_dir
        self.logs_dir = self.user_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # === 日志写入 ===

    def _write_log(self, category: str, message: str):
        """写入分类日志文件。"""
        ym = datetime.now().strftime("%Y%m")
        log_path = self.logs_dir / f"{category}_{ym}.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {message}\n")

    def log_search(self, query: str, sources: list[str],
                   num_results: int, num_selected: int, duration_sec: float):
        msg = f"SEARCH | user={self.username} | query=\"{query}\" | sources={','.join(sources)} | found={num_results} | selected={num_selected} | duration={duration_sec:.1f}s"
        self._write_log("search", msg)
        self.storage.log_search(query, sources, num_results, num_selected, duration_sec)

    def log_upload(self, file_name: str, title: str, paper_id: int,
                   status: str = "ok", error: str = "",
                   *, abstract: str = "", core_claim: str = ""):
        """记录论文入库操作。

        Args:
            abstract: 论文摘要（首 200 字），写入 summary 供 LLM 生成会话摘要时使用
            core_claim: 核心结论，同上
        """
        msg = f"UPLOAD | user={self.username} | file=\"{file_name}\" | title=\"{title}\" | status={status} | paper_id={paper_id}"
        if error:
            msg += f" | error={error}"
        self._write_log("upload", msg)

        # summary 包含可供 LLM 理解的内容，而非仅标题
        parts = [f"入库: {title}"]
        if core_claim:
            parts.append(f"核心结论: {core_claim[:200]}")
        elif abstract:
            parts.append(f"摘要: {abstract[:200]}")
        summary = "。".join(parts)

        self.storage.log_operation("upload", summary,
                                   details={"file": file_name, "title": title,
                                            "paper_id": paper_id, "status": status,
                                            "core_claim": core_claim[:200],
                                            "abstract": abstract[:200]})

    def log_review(self, topic: str, kb_papers: int, new_papers: int,
                   word_count: int, duration_sec: float,
                   *, conclusion: str = "", key_findings: str = ""):
        """记录文献综述操作。

        Args:
            conclusion: 综述的核心结论或摘要（首 300 字），写入 summary
            key_findings: 关键发现，同上
        """
        msg = f"REVIEW | user={self.username} | topic=\"{topic}\" | kb_papers={kb_papers} | new_papers={new_papers} | word_count={word_count} | duration={duration_sec:.1f}s"
        self._write_log("review", msg)

        parts = [f"文献综述: {topic}"]
        if conclusion:
            parts.append(f"核心结论: {conclusion[:300]}")
        if key_findings:
            parts.append(f"关键发现: {key_findings[:200]}")
        summary = "。".join(parts)

        self.storage.log_operation("review", summary,
                                   details={"topic": topic, "kb_papers": kb_papers,
                                            "new_papers": new_papers, "word_count": word_count,
                                            "conclusion": conclusion[:300]},
                                   duration_sec=duration_sec)

    def log_progress(self, topic: str, entry_type: str, title: str,
                      *, content: str = "", results: str = "",
                      insights: str = "", next_actions: str = ""):
        """记录研究进展操作。

        Args:
            content: 详细描述
            results: 实验数据/结果
            insights: 获得的洞察
            next_actions: 下一步计划
        """
        msg = f"PROGRESS | user={self.username} | topic=\"{topic}\" | type={entry_type} | title=\"{title}\""
        self._write_log("progress", msg)

        parts = [f"[{entry_type}] {title}"]
        if content:
            parts.append(f"描述: {content[:200]}")
        if results:
            parts.append(f"结果: {results[:200]}")
        if insights:
            parts.append(f"洞察: {insights[:150]}")
        if next_actions:
            parts.append(f"下一步: {next_actions[:150]}")
        summary = "。".join(parts)

        self.storage.log_operation("progress", summary,
                                   details={"topic": topic, "type": entry_type,
                                            "content": content[:200],
                                            "results": results[:200],
                                            "insights": insights[:150]})

    def log_system(self, level: str, message: str):
        msg = f"{level} | user={self.username} | {message}"
        self._write_log("system", msg)

    def log_backup(self, action: str, files: int = 0, status: str = "ok"):
        msg = f"BACKUP | {action} | files={files} | status={status}"
        self._write_log("backup", msg)

    # === 备份操作 ===

    def full_backup(self) -> str:
        """执行全量 JSON 备份。

        Returns:
            备份文件路径的描述
        """
        try:
            self.storage.write_backup()
            # 统计
            today = datetime.now().strftime("%Y%m%d")
            backup_files = list(self.storage.backup_dir.glob(f"*_{today}.json"))
            self.log_backup("full_backup", files=len(backup_files))
            return f"备份完成: {len(backup_files)} 个文件 → {self.storage.backup_dir}"
        except Exception as e:
            self.log_backup("full_backup", status="failed")
            self.log_system("ERROR", f"备份失败: {e}")
            return f"备份失败: {e}"

    def verify_integrity(self) -> dict:
        """启动时验证数据完整性。

        Returns:
            {"ok": bool, "issues": ["问题描述"...]}
        """
        issues = []

        # 检查 SQLite
        if not (self.user_dir / "library.db").exists():
            issues.append("SQLite 库不存在")
        else:
            try:
                cnt = self.storage.count_papers()
            except Exception as e:
                issues.append(f"SQLite 读取异常: {e}")

        # 检查备份
        backups = list(self.storage.backup_dir.glob("*.json"))
        if not backups:
            issues.append("无 JSON 备份（新用户正常）")
        else:
            # 找最新备份
            latest = max(backups, key=lambda p: p.stat().st_mtime)
            age_days = (datetime.now().timestamp() - latest.stat().st_mtime) / 86400
            if age_days > 7:
                issues.append(f"最新备份 {age_days:.0f} 天前")

        # 检查 Qdrant 连接
        try:
            from ..rag.vector_store import VectorStore
            vs = VectorStore()
            vs.ensure_collection(self.username, "papers")
        except Exception as e:
            issues.append(f"Qdrant 连接失败: {e}")

        ok = len(issues) == 0
        if not ok:
            self.log_system("WARN" if len(issues) < 3 else "ERROR",
                           f"数据完整性检查: {'; '.join(issues)}")

        return {"ok": ok, "issues": issues}

    def rebuild_vectors_from_sqlite(self) -> str:
        """从 SQLite 重建 Qdrant 向量索引（灾难恢复）。

        读 SQLite 中所有论文 → 重新嵌入 → 重新写入 Qdrant。
        """
        papers = self.storage.get_all_papers()
        if not papers:
            return "SQLite 中无论文，无需重建。"

        from ..rag.vector_store import VectorStore
        from ..rag.ingestion import IngestionPipeline

        vs = VectorStore()
        pipeline = IngestionPipeline(self.storage, self.username)

        # 清空旧 collection
        vs.delete_collection(self.username, "papers")
        vs.ensure_collection(self.username, "papers")

        rebuilt = 0
        for paper in papers:
            try:
                annotation = paper.get("annotation")
                if isinstance(annotation, str):
                    annotation = json.loads(annotation)
                file_path = paper.get("file_path", "")
                full_text = ""
                if file_path and os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        full_text = f.read()

                if full_text:
                    chunks = pipeline.chunk_paper(full_text, paper["id"], annotation or {})
                else:
                    chunks = pipeline.chunk_summary_only(paper, paper["id"], annotation or {})

                if chunks:
                    vs.upsert(self.username, "papers", chunks)
                rebuilt += 1
            except Exception as e:
                self.log_system("ERROR", f"重建 paper_id={paper.get('id')} 失败: {e}")

        self.log_system("INFO", f"向量重建完成: {rebuilt}/{len(papers)} 篇论文")
        self.log_backup("rebuild_vectors", files=rebuilt)
        return f"向量重建完成: {rebuilt}/{len(papers)} 篇论文"
