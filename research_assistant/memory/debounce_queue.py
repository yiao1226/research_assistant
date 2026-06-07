"""防抖记忆更新队列 — 参考 DeerFlow 的 MemoryUpdateQueue。

核心机制:
  1. 每次 add_turn 后启动/重置定时器（默认 30 秒）
  2. 同一 session 的新条目覆盖旧条目（因为 working memory 的 turns 是累积的完整历史）
  3. 定时器触发时执行 LLM 事实提取 + 置信度评估
  4. add_nowait() 提供紧急刷盘通道（退出前/备份前）

设计要点（来自 DeerFlow 的讨论）:
  - 覆盖而非追加：self.turns 已经是累积的完整对话历史，每次 add 都是"当前全貌的快照"
  - 30 秒不是精确值，目的是等用户把一轮话说完整
  - threading.Timer 在独立线程执行，不阻塞主流程
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_DEBOUNCE_SECONDS = 30.0


@dataclass
class MemoryQueueEntry:
    """队列条目：一次待处理的记忆更新。"""

    session_id: str
    """会话标识（通常为 username，每个用户独立队列）"""

    callback: Callable
    """到期时调用的函数。签名: callback() -> None"""

    created_at: datetime = field(default_factory=datetime.now)
    """入队时间戳"""

    correction_detected: bool = False
    reinforcement_detected: bool = False


class MemoryUpdateQueue:
    """带防抖机制的记忆更新队列。

    用法:
        queue = MemoryUpdateQueue(debounce_seconds=30.0)

        # 每次对话轮次后调用
        queue.add("user_A", my_consolidate_func)

        # 紧急刷盘（退出前）
        queue.flush_nowait()
    """

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS):
        self._debounce_seconds = debounce_seconds
        self._queue: dict[str, MemoryQueueEntry] = {}  # session_id -> entry
        self._timers: dict[str, threading.Timer] = {}   # session_id -> timer
        self._lock = threading.Lock()
        self._processing = False

    # ── 公共 API ──

    def add(
        self,
        session_id: str,
        callback: Callable,
        *,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """添加/更新记忆更新任务。

        同一 session_id 的新调用会覆盖旧条目并重置定时器。
        这是设计意图——LangGraph 的 state['messages'] 是累积的完整历史。

        Args:
            session_id: 会话标识
            callback: 到期时执行的回调（在后台线程中同步调用）
            correction_detected: 本轮是否检测到纠正信号
            reinforcement_detected: 本轮是否检测到认可信号
        """
        with self._lock:
            # 合并信号：如果任何一次检测到纠正/认可，标记保持为 True
            existing = self._queue.get(session_id)
            merged_correction = correction_detected or (
                existing.correction_detected if existing else False
            )
            merged_reinforcement = reinforcement_detected or (
                existing.reinforcement_detected if existing else False
            )

            self._queue[session_id] = MemoryQueueEntry(
                session_id=session_id,
                callback=callback,
                correction_detected=merged_correction,
                reinforcement_detected=merged_reinforcement,
            )
            self._reset_timer_locked(session_id)

        logger.debug(
            "记忆更新已入队: session=%s queue_size=%d",
            session_id, len(self._queue),
        )

    def add_nowait(self, session_id: str, callback: Callable) -> None:
        """紧急添加：跳过防抖，立即在后台线程处理。

        用于: 用户退出、手动备份、会话被压缩前。
        """
        with self._lock:
            self._queue[session_id] = MemoryQueueEntry(
                session_id=session_id,
                callback=callback,
            )
            self._schedule_timer_locked(session_id, delay_seconds=0)

        logger.info("记忆更新已入队（立即处理）: session=%s", session_id)

    def flush_nowait(self) -> None:
        """立即处理所有待处理条目（后台线程）。"""
        with self._lock:
            # 取消所有定时器
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            # 立即调度
            entries = list(self._queue.values())
            self._queue.clear()

        if entries:
            logger.info("冲刷 %d 个待处理记忆更新", len(entries))
            for entry in entries:
                try:
                    entry.callback()
                except Exception:
                    logger.exception("记忆更新失败: session=%s", entry.session_id)

    def cancel(self, session_id: str) -> None:
        """取消指定会话的待处理更新。"""
        with self._lock:
            if session_id in self._timers:
                self._timers[session_id].cancel()
                del self._timers[session_id]
            self._queue.pop(session_id, None)

    # ── 内部方法 ──

    def _reset_timer_locked(self, session_id: str) -> None:
        """重置防抖定时器（需持有锁）。"""
        self._schedule_timer_locked(session_id, self._debounce_seconds)

    def _schedule_timer_locked(self, session_id: str, delay_seconds: float) -> None:
        """调度定时器（需持有锁）。

        如果 delay_seconds=0，立即在后台线程执行。
        """
        # 取消旧定时器
        old_timer = self._timers.pop(session_id, None)
        if old_timer is not None:
            old_timer.cancel()

        if delay_seconds <= 0:
            # 立即处理：启动后台线程
            entry = self._queue.pop(session_id, None)
            if entry is not None:
                t = threading.Thread(
                    target=self._process_entry,
                    args=(entry,),
                    daemon=True,
                    name=f"memory-flush-{session_id}",
                )
                t.start()
        else:
            timer = threading.Timer(delay_seconds, self._on_timer_fire, args=(session_id,))
            timer.daemon = True
            timer.start()
            self._timers[session_id] = timer

    def _on_timer_fire(self, session_id: str) -> None:
        """定时器到期回调（在 Timer 线程中执行）。"""
        with self._lock:
            self._timers.pop(session_id, None)
            entry = self._queue.pop(session_id, None)

        if entry is None:
            return

        self._process_entry(entry)

    def _process_entry(self, entry: MemoryQueueEntry) -> None:
        """执行单个记忆更新条目。"""
        logger.info("处理记忆更新: session=%s correction=%s reinforcement=%s",
                     entry.session_id, entry.correction_detected,
                     entry.reinforcement_detected)
        try:
            entry.callback()
            logger.info("记忆更新完成: session=%s", entry.session_id)
        except Exception:
            logger.exception("记忆更新失败: session=%s", entry.session_id)

    # ── 状态查询 ──

    @property
    def pending_count(self) -> int:
        """待处理的条目数。"""
        with self._lock:
            return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """是否有正在处理的条目。"""
        return self._processing


# ── 全局单例 ──

_global_queue: MemoryUpdateQueue | None = None
_queue_lock = threading.Lock()


def get_memory_queue(debounce_seconds: float | None = None) -> MemoryUpdateQueue:
    """获取全局记忆更新队列单例。

    Args:
        debounce_seconds: 仅首次创建时生效，默认 30 秒
    """
    global _global_queue
    with _queue_lock:
        if _global_queue is None:
            delay = (
                debounce_seconds
                if debounce_seconds is not None
                else DEFAULT_DEBOUNCE_SECONDS
            )
            _global_queue = MemoryUpdateQueue(debounce_seconds=delay)
        return _global_queue


def reset_memory_queue() -> None:
    """重置全局队列（测试用）。"""
    global _global_queue
    with _queue_lock:
        if _global_queue is not None:
            _global_queue.flush_nowait()
        _global_queue = None
