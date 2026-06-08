"""工作记忆 — Q&A 多轮对话上下文（会话级，内存存储）。

改进版（参考 DeerFlow 记忆系统）:
  P0-1: 防抖队列 — 30秒防抖替代固定10轮触发，不丢中途退出数据
  P1-1: 结构化事实提取 — LLM 提取 {content, category, confidence} 三元组
  P1-2: 纠正/认可信号检测 — 用户说"不对"触发优先修正

原设计:
  - 纯内存 + 容量上限 10 轮
  - 超出时压缩最旧的 5 轮为摘要
  - 每轮存储: 问题 + 答案 + 引用论文 + 时间戳

新增:
  - add_turn 后启动防抖定时器（默认30秒）
  - 定时器到期 → LLM 提取结构化事实 → 写入 UserProfileManager
  - 用户退出时 flush_nowait() 保证不丢数据
  - add_nowait 紧急通道（会话压缩前刷盘）
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from ..utils import get_llm

logger = logging.getLogger(__name__)

MAX_TURNS = 10          # 最大保留轮数
EVICT_BATCH = 5         # 超出时每次压缩的轮数
DEFAULT_DEBOUNCE = 30.0 # 防抖秒数（可配置）

CONSOLIDATE_PROMPT = """将以下 Q&A 对话压缩为简短摘要（1-2句话）。
提取: 讨论了什么主题？得到了什么关键结论？引用了哪些论文？

对话:
{turns_text}

只输出摘要文本，不要 JSON。"""


class QATurn:
    """单轮 Q&A 记录。"""

    def __init__(self, question: str, answer: str,
                 cited_papers: list[dict] | None = None,
                 understanding: str = "",
                 correction_detected: bool = False,
                 reinforcement_detected: bool = False):
        self.question = question
        self.answer = answer
        self.cited_papers = cited_papers or []
        self.understanding = understanding
        self.timestamp = datetime.now()
        self.correction_detected = correction_detected
        self.reinforcement_detected = reinforcement_detected

    def to_text(self) -> str:
        papers_str = ", ".join(
            p.get("title", "?")[:40] for p in self.cited_papers[:3]
        ) or "无"
        return (
            f"Q: {self.question}\n"
            f"A: {self.answer[:300]}\n"
            f"引用: {papers_str}"
        )


class WorkingMemory:
    """会话级工作记忆（改进版）。

    改进:
      - 防抖队列替代固定轮次触发
      - 结构化事实提取 + 置信度
      - 纠正/认可信号检测
    """

    def __init__(self, username: str, user_dir: str = "",
                 *, debounce_seconds: float = DEFAULT_DEBOUNCE,
                 storage=None):
        self.username = username
        self.user_dir = user_dir  # PerUserStorage.user_dir，用于 episodic 持久化
        self._external_storage = storage  # 外部传入的 PerUserStorage（避免重复创建）
        self.turns: list[QATurn] = []
        self._consolidated: list[str] = []  # 已压缩的历史摘要
        self._episodic = None  # 延迟加载
        self._profile_manager = None  # 延迟加载
        self._debounce_seconds = debounce_seconds

        # 信号追踪（跨轮次累积）
        self._any_correction_detected = False
        self._any_reinforcement_detected = False

        # 事实提取进度追踪：避免重复发送已提取的轮次给 LLM
        self._last_extracted_index = 0  # 上次提取到了 turns 的第几个位置

        # 防抖定时器
        self._debounce_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

    # ── 延迟加载 ──

    @property
    def episodic(self):
        """延迟加载情景记忆（优先复用外部传入的 storage，避免重复创建 PerUserStorage）。"""
        if self._episodic is None:
            from ..memory.episodic import EpisodicMemory
            if self._external_storage is not None:
                storage = self._external_storage
            else:
                from ..core.storage import PerUserStorage
                from pathlib import Path
                if self.user_dir:
                    user_dir = Path(self.user_dir)
                else:
                    user_dir = Path("./data/users") / self.username
                storage = PerUserStorage(user_dir)
            self._episodic = EpisodicMemory(storage, self.username)
        return self._episodic

    @property
    def profile_manager(self):
        """延迟加载用户画像管理器。"""
        if self._profile_manager is None:
            from ..memory.user_profile import UserProfileManager
            self._profile_manager = UserProfileManager(self.username, self.user_dir)
        return self._profile_manager

    # ── 核心操作 ──

    def add_turn(self, question: str, answer: str,
                 cited_papers: list[dict] | None = None,
                 understanding: str = ""):
        """添加一轮 Q&A。

        改进后的流程:
          1. 检测纠正/认可信号
          2. 添加到 turns 列表
          3. 超过 MAX_TURNS 时触发旧轮次压缩（摘要）
          4. 启动/重置防抖定时器 → 到期后 LLM 提取结构化事实
        """
        # 信号检测
        from .message_processing import classify_turn
        signals = classify_turn(question)
        self._any_correction_detected = self._any_correction_detected or signals["correction_detected"]
        self._any_reinforcement_detected = self._any_reinforcement_detected or signals["reinforcement_detected"]

        turn = QATurn(
            question, answer, cited_papers, understanding,
            correction_detected=signals["correction_detected"],
            reinforcement_detected=signals["reinforcement_detected"],
        )
        self.turns.append(turn)

        # 硬上限保护：超过 MAX_TURNS 时压缩旧轮次
        if len(self.turns) > MAX_TURNS:
            self._consolidate_oldest()

        # 防抖：每次新的对话轮次后重置定时器
        self._reset_debounce_timer()

    def add_nowait_flush(self):
        """紧急刷盘：立即提取事实并写入持久存储。

        用于: 用户退出、手动备份、会话压缩前。
        不等防抖定时器，直接调用 LLM 提取。
        """
        self._cancel_timer()
        self._extract_and_persist_facts()

    # ── 防抖定时器 ──

    def _reset_debounce_timer(self):
        """重置防抖定时器。

        每次 add_turn 后调用。新轮次覆盖旧定时器——
        因为 self.turns 已经是累积的完整历史，不是增量。
        """
        with self._timer_lock:
            # 取消旧定时器
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()

            # 启动新定时器
            self._debounce_timer = threading.Timer(
                self._debounce_seconds,
                self._on_debounce_fire,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()
            logger.debug(
                "防抖定时器已重置: %ss, turns=%d",
                self._debounce_seconds, len(self.turns),
            )

    def _cancel_timer(self):
        """取消定时器（退出/刷盘前）。"""
        with self._timer_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None

    def _on_debounce_fire(self):
        """定时器到期回调（在 Timer 线程中执行）。

        执行 LLM 事实提取 + 持久化写入。
        """
        logger.info("防抖定时器到期，开始提取事实: turns=%d", len(self.turns))
        self._extract_and_persist_facts()

    def _extract_and_persist_facts(self):
        """LLM 提取结构化事实并持久化。

        只发送上次提取之后的新轮次，避免重复传输已处理过的对话文本。
        在后台线程中同步执行（不阻塞主流程）。
        """
        if len(self.turns) < 2:
            logger.debug("轮次不足，跳过事实提取")
            return

        # ── 只取未提取过的新轮次 ──
        new_turns = self.turns[self._last_extracted_index:]
        if len(new_turns) < 2:
            logger.debug("新轮次不足（%d），跳过事实提取", len(new_turns))
            return

        try:
            from .fact_extraction import extract_facts_from_conversation

            # 加载已有事实
            try:
                existing_facts = self.profile_manager.get_facts_list()
            except Exception:
                existing_facts = []

            # LLM 提取（只传新轮次）
            result = extract_facts_from_conversation(
                new_turns,
                existing_facts=existing_facts,
                correction_detected=self._any_correction_detected,
                reinforcement_detected=self._any_reinforcement_detected,
            )

            new_facts = result.get("newFacts", [])
            user_context = result.get("userContext", {})
            facts_to_remove = result.get("factsToRemove", [])

            if new_facts or user_context:
                self.profile_manager.update_facts(
                    user_context=user_context,
                    new_facts=new_facts,
                    facts_to_remove=facts_to_remove,
                )
                logger.info(
                    "事实提取完成: new=%d remove=%d ctx=%s",
                    len(new_facts), len(facts_to_remove),
                    bool(user_context.get("researchFocus")),
                )
                # 记录操作日志
                try:
                    self.episodic.storage.log_operation(
                        "fact_extraction",
                        f"LLM 提取 {len(new_facts)} 条事实, "
                        f"删除 {len(facts_to_remove)} 条, "
                        f"纠正={self._any_correction_detected}, "
                        f"认可={self._any_reinforcement_detected}",
                        details={
                            "new_count": len(new_facts),
                            "removed_count": len(facts_to_remove),
                            "correction": self._any_correction_detected,
                            "reinforcement": self._any_reinforcement_detected,
                        },
                    )
                except Exception:
                    logger.debug("操作日志记录失败", exc_info=True)
            else:
                logger.debug("事实提取无新结果")

            # 重置信号 + 更新提取进度
            self._any_correction_detected = False
            self._any_reinforcement_detected = False
            self._last_extracted_index = len(self.turns)   # ← 标记：这些轮次已提取

        except Exception:
            logger.exception("事实提取失败")

    # ── 原有功能（保持向后兼容）──

    def get_context(self, n_turns: int = 5) -> str:
        """获取最近 N 轮对话上下文，用于注入 LLM prompt。

        Returns:
            格式化的对话历史文本，可直接拼入 system/user prompt
        """
        if not self.turns and not self._consolidated:
            return ""

        parts = []

        # 历史摘要（压缩的旧对话）
        if self._consolidated:
            parts.append("## 历史讨论摘要")
            for i, summary in enumerate(self._consolidated, 1):
                parts.append(f"{i}. {summary}")

        # 最近 N 轮完整对话
        recent = self.turns[-n_turns:] if n_turns > 0 else self.turns
        if recent:
            parts.append("\n## 最近对话")
            for i, turn in enumerate(recent, 1):
                parts.append(f"--- 第{i}轮 ---")
                parts.append(turn.to_text())

        return "\n".join(parts)

    def get_last_question(self) -> str:
        """获取上一轮的问题（用于追问检测）。"""
        if self.turns:
            return self.turns[-1].question
        return ""

    def get_cited_papers(self) -> list[dict]:
        """获取本轮会话中引用过的所有论文（去重）。"""
        seen = set()
        papers = []
        for turn in self.turns:
            for p in turn.cited_papers:
                key = p.get("arxiv_id") or p.get("doi") or p.get("title")
                if key and key not in seen:
                    seen.add(key)
                    papers.append(p)
        return papers

    # ── Consolidate: 压缩旧对话为摘要 ──

    def _consolidate_oldest(self):
        """压缩最旧的 EVICT_BATCH 轮对话，摘要记入操作日志。"""
        if len(self.turns) <= EVICT_BATCH:
            return

        oldest = self.turns[:EVICT_BATCH]
        turns_text = "\n\n".join(t.to_text() for t in oldest)

        try:
            llm = get_llm(temperature=0.1, max_tokens=256)
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content="你是对话摘要专家。"),
                HumanMessage(
                    content=CONSOLIDATE_PROMPT.format(turns_text=turns_text)
                ),
            ])
            summary = str(response.content).strip()
            if summary:
                self._consolidated.append(summary)
                # 记录操作日志
                try:
                    self.episodic.storage.log_operation(
                        "qa_consolidate",
                        f"Q&A 压缩摘要: {summary[:100]}",
                        details={"summary": summary, "turn_count": EVICT_BATCH},
                    )
                except Exception:
                    logger.debug("操作日志记录失败", exc_info=True)
        except Exception:
            logger.debug("工作记忆压缩失败", exc_info=True)
            # 降级：不压缩，直接丢弃最旧的
            summary = f"（已丢弃 {EVICT_BATCH} 轮对话）"
            self._consolidated.append(summary)

        # 移除已压缩的轮次，同步调整事实提取进度指针
        self.turns = self.turns[EVICT_BATCH:]
        self._last_extracted_index = max(0, self._last_extracted_index - EVICT_BATCH)

    def get_session_summary(self) -> dict:
        """生成整个会话的摘要（退出时调用）。

        Returns:
            适合传给 EpisodicMemory.save_session_summary() 的 dict
        """
        topics = set()
        paper_ids = []
        for turn in self.turns:
            # 从 Phase 1 意图识别结果提取话题（零额外LLM成本）
            if turn.understanding:
                topics.add(turn.understanding)
            for p in turn.cited_papers:
                pid = p.get("arxiv_id") or p.get("doi") or p.get("title")
                if pid:
                    paper_ids.append(pid)

        return {
            "topics": list(topics),
            "papers_added": 0,
            "papers_analyzed": len(set(paper_ids)),
            "progress_entries": 0,
            "key_discussions": [
                t.question[:80] for t in self.turns[-3:]
            ],
            "unresolved_questions": [],
            "plan_updates": "",
            "suggested_followups": [],
            "session_date": datetime.now().strftime("%Y-%m-%d"),
        }

    def clear(self):
        """清空工作记忆。退出前应调用 add_nowait_flush()。"""
        self._cancel_timer()
        self.turns.clear()
        self._consolidated.clear()
        self._any_correction_detected = False
        self._any_reinforcement_detected = False
        self._last_extracted_index = 0

    def stats(self) -> dict:
        return {
            "active_turns": len(self.turns),
            "consolidated_summaries": len(self._consolidated),
            "cited_papers": len(self.get_cited_papers()),
            "correction_detected": self._any_correction_detected,
            "reinforcement_detected": self._any_reinforcement_detected,
            "debounce_seconds": self._debounce_seconds,
        }
