"""跨会话情景记忆 — 会话摘要生成、存储、召回。

每会话结束时:
  1. LLM 阅读当前会话操作记录 → 生成摘要
  2. 用户预览/编辑摘要
  3. 摘要嵌入 → Qdrant memory collection
  4. 摘要写入 SQLite session_log 表
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.storage import PerUserStorage
from ..rag.vector_store import VectorStore
from ..utils import get_llm, extract_json_from_llm_response

SESSION_SUMMARY_PROMPT = """从以下操作日志中总结本次研究会话的要点。

操作日志:
{log_text}

请以 JSON 格式输出:
{{
  "topics": ["本次讨论了哪些主题"],
  "papers_added": 入库论文数,
  "papers_analyzed": 分析论文数,
  "progress_entries": 进展记录数,
  "key_discussions": ["1-3条关键讨论/发现"],
  "unresolved_questions": ["未解决的问题"],
  "plan_updates": "研究计划有什么变化（如有）",
  "suggested_followups": ["建议下一步做什么"]
}}

只返回 JSON。"""


class EpisodicMemory:
    """跨会话情景记忆管理。"""

    def __init__(self, storage: PerUserStorage, username: str):
        self.storage = storage
        self.username = username
        self.vector_store = VectorStore()

    def generate_session_summary(self) -> dict:
        """从今日操作日志生成会话摘要。"""
        import sqlite3
        today = datetime.now().strftime("%Y-%m-%d")

        # 读取今天的操作日志
        with self.storage._conn() as c:
            rows = c.execute(
                "SELECT * FROM session_log WHERE date(timestamp)=? ORDER BY timestamp",
                (today,),
            ).fetchall()
            if not rows:
                return {"topics": [], "papers_added": 0}

            log_text = "\n".join(
                f"[{r['timestamp']}] {r['op_type']}: {r['summary']}"
                for r in rows
            )

        llm = get_llm(temperature=0.2, max_tokens=1024)

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研会话总结专家。只返回 JSON。"),
                HumanMessage(content=SESSION_SUMMARY_PROMPT.format(log_text=log_text[:3000])),
            ])
            summary = extract_json_from_llm_response(str(response.content))
        except (json.JSONDecodeError, AttributeError):
            summary = {
                "topics": [],
                "papers_added": 0,
                "papers_analyzed": 0,
                "progress_entries": 0,
                "key_discussions": [],
                "unresolved_questions": [],
                "plan_updates": "",
                "suggested_followups": [],
            }

        summary["session_date"] = today
        return summary

    def save_session_summary(self, summary: dict):
        """保存会话摘要到 Qdrant + SQLite。

        Args:
            summary: generate_session_summary 的输出（可能已被用户编辑）
        """
        text_parts = [
            f"主题: {'、'.join(summary.get('topics', []))}",
            f"关键讨论: {'; '.join(summary.get('key_discussions', []))}",
            f"未解决问题: {'; '.join(summary.get('unresolved_questions', []))}",
            f"下一步建议: {'; '.join(summary.get('suggested_followups', []))}",
            summary.get("plan_updates", ""),
        ]
        text = " | ".join(p for p in text_parts if p)
        if not text:
            return

        date_str = summary.get('session_date', datetime.now().strftime('%Y-%m-%d'))
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"session_{date_str}"))

        self.vector_store.upsert(self.username, "memory", [{
            "id": session_id,
            "text": text,
            "payload": {
                "type": "session_summary",
                "session_date": summary.get("session_date", ""),
                "topics": json.dumps(summary.get("topics", []), ensure_ascii=False),
                "papers_added": summary.get("papers_added", 0),
                "progress_entries": summary.get("progress_entries", 0),
            },
        }])

        # 同时写入 session_log
        self.storage.log_operation(
            "session_summary",
            f"会话摘要: {summary.get('session_date', '')}",
            details=summary,
        )

    def recall_context(self, query: str, limit: int = 5) -> list[dict]:
        """语义召回相关历史记忆。"""
        return self.vector_store.search(self.username, "memory", query, limit=limit)

    def recall_context_hybrid(
        self, query: str, limit: int = 5,
        *, decay_days: float = 30.0, fetch_limit: int = 20,
    ) -> list[dict]:
        """混合检索：语义相似度 × 时间衰减。

        避免纯语义检索的偏科——"昨天讨论的 CVD 温度"和"两个月前
        讨论的 CVD 温度"语义分数可能相近，但昨天的显然更有价值。
        用指数衰减压低旧会话的得分。

        公式: combined_score = semantic_score × e^(-days_ago / decay_days)

        Args:
            query: 搜索文本
            limit: 最终返回条数
            decay_days: 衰减半衰期（30天前的会话权重降为 ~0.37）
            fetch_limit: 从 Qdrant 取多少候选再混合排序

        Returns:
            按 combined_score 降序排列的结果，格式同 search()
        """
        from datetime import datetime
        import math

        # 多拿一些候选，后续用时间加权重排
        candidates = self.vector_store.search(
            self.username, "memory", query, limit=fetch_limit,
        )
        if not candidates:
            return []

        now = datetime.now()
        scored = []
        for item in candidates:
            # 提取日期
            payload = item.get("payload", {})
            date_str = payload.get("session_date", "")
            try:
                session_date = datetime.strptime(date_str, "%Y-%m-%d")
                days_ago = (now - session_date).days
            except (ValueError, TypeError):
                days_ago = 365  # 没有日期信息的放最后

            # 指数衰减：30 天前 → 0.37x，60 天前 → 0.14x
            time_weight = math.exp(-days_ago / decay_days) if days_ago >= 0 else 1.0
            semantic_score = item.get("score", 0.0)
            combined_score = semantic_score * time_weight

            scored.append({
                **item,
                "text": item.get("payload", {}).get("text", ""),
                "semantic_score": round(semantic_score, 4),
                "time_weight": round(time_weight, 4),
                "combined_score": round(combined_score, 4),
                "days_ago": days_ago,
            })

        # 按综合得分重排
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:limit]

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        """获取最近的会话摘要。"""
        import sqlite3
        with self.storage._conn() as c:
            rows = c.execute(
                "SELECT * FROM session_log WHERE op_type='session_summary' ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self.storage._row_to_dict(r) for r in rows]

    def get_unresolved_questions(self, limit: int = 10) -> list[str]:
        """收集所有未解决的问题。"""
        recent = self.get_recent_sessions(limit=limit)
        questions = []
        for s in recent:
            details = s.get("details", {})
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except json.JSONDecodeError:
                    details = {}
            for q in details.get("unresolved_questions", []):
                if q not in questions:
                    questions.append(q)
        return questions
