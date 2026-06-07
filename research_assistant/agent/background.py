"""后台任务管理器 — thread pool + SQLite 持久化 + 优雅关闭。

参考 Claude Code s13 的 background_tasks 设计:
  - 慢操作 → daemon 线程 + 占位符 ToolResult
  - 结果通过 before_llm Hook 注入到下一轮对话
  - 退出时 save state → 下次登录可恢复

生产级改进:
  - SQLite 持久化: 崩溃不丢任务
  - ThreadPoolExecutor: 控制并发上限
  - shutdown(timeout): 优雅关闭 + 中断态存盘
  - 双重 Ctrl+C: 第一轮优雅 → 第二轮强杀
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)

# ── 慢操作识别 ──

# 注意: search_papers_online 不列为慢操作——它是交互式的，用户主动等结果，
# 走后台会破坏搜索→展示→交互选择的流程。只有纯计算/嵌入类操作走后台。
SLOW_TOOLS: dict[str, float] = {
    "ingest_papers": 60.0,           # PDF 入库 + BGE 嵌入，大文档 ~60s
}

# 默认优雅关闭等待时间（秒）
DEFAULT_SHUTDOWN_TIMEOUT = 10.0

# 线程池大小：同类型工具最多同时跑的并发数
MAX_CONCURRENT_INGEST = 1    # BGE 嵌入 CPU 密集，单线程更好


def _is_slow_tool(tool_name: str) -> bool:
    """判断是否为慢操作。"""
    return tool_name in SLOW_TOOLS


def _max_concurrent(tool_name: str) -> int:
    """返回该工具类型的最大并发数。"""
    if tool_name == "ingest_papers":
        return MAX_CONCURRENT_INGEST
    return 1


def _timeout_for(tool_name: str) -> float:
    """返回该工具类型的合理超时时间。"""
    return SLOW_TOOLS.get(tool_name, 30.0)


# ═══════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════

@dataclass
class BackgroundTask:
    """单个后台任务的状态。"""
    task_id: str
    tool_name: str
    command: str              # 人类可读描述，如 "search_papers_online(...)"
    status: str               # running | completed | failed | interrupted
    started_at: str
    result: str = ""
    error: str = ""
    finished_at: str = ""
    _future: Future | None = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════
# 后台任务管理器
# ═══════════════════════════════════════════════════════════

class BackgroundTaskManager:
    """后台任务生命周期管理。

    用法:
        mgr = BackgroundTaskManager(storage, username)

        # 派发后台任务
        task_id = mgr.dispatch("search_papers_online",
                              "search_papers_online('CVD temperature')",
                              lambda: do_search())

        # 每轮 Agent 循环前收集通知（通过 Hook）
        notifications = mgr.collect_notifications()

        # 退出时优雅关闭
        mgr.shutdown(timeout=10.0)
    """

    def __init__(self, storage, username: str):
        self._storage = storage
        self._username = username

        # 按工具类型分组线程池，避免慢操作互相阻塞
        self._executors: dict[str, ThreadPoolExecutor] = {}

        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._shutting_down = False

        # Ctrl+C 计数器（双重检测）
        self._interrupt_count = 0

    # ── 线程池管理 ──

    def _get_executor(self, tool_name: str) -> ThreadPoolExecutor:
        """获取或创建该工具类型的线程池。"""
        if tool_name not in self._executors:
            self._executors[tool_name] = ThreadPoolExecutor(
                max_workers=_max_concurrent(tool_name),
                thread_name_prefix=f"bg_{tool_name}",
            )
        return self._executors[tool_name]

    # ── 派发后台任务 ──

    def dispatch(self, tool_name: str, command: str,
                 fn: Callable[[], str]) -> str:
        """派发后台任务，立即返回 task_id。

        Args:
            tool_name: 工具名称（决定线程池和超时策略）
            command: 人类可读的描述，用于恢复提示
            fn: 实际执行的函数，返回字符串结果

        Returns:
            task_id: 形如 "bg_0001" 的后台任务 ID
        """
        if self._shutting_down:
            return _make_blocked_id(tool_name, "系统正在关闭，不接受新任务")

        self._counter += 1
        task_id = f"bg_{self._counter:04d}"

        task = BackgroundTask(
            task_id=task_id,
            tool_name=tool_name,
            command=command,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        # 持久化: 写入 SQLite
        try:
            self._storage.save_background_task(
                task_id, tool_name, command, self._username,
            )
        except Exception:
            logger.debug("后台任务持久化失败（存储不可用）", exc_info=True)

        def worker():
            try:
                result = fn()
                with self._lock:
                    task.result = result
                    task.status = "completed"
                    task.finished_at = datetime.now().isoformat()
                self._persist(task)
                # 主动通知：后台任务完成后立即打印，不等下一轮 LLM
                import sys
                summary = result[:200].replace("\n", " ") + ("..." if len(result) > 200 else "")
                print(f"\n✅ 后台任务完成: {command[:60]}", flush=True)
                print(f"   {summary}", flush=True)
                print(f"   💡 输入追问或继续对话即可看到结果", flush=True)
            except Exception as e:
                with self._lock:
                    task.error = str(e)
                    task.status = "failed"
                    task.finished_at = datetime.now().isoformat()
                self._persist(task)
                import sys
                print(f"\n❌ 后台任务失败: {command[:60]} — {e}", flush=True)

        executor = self._get_executor(tool_name)
        future = executor.submit(worker)
        task._future = future

        with self._lock:
            self._tasks[task_id] = task

        logger.info("后台任务派发: %s → %s", task_id, command[:60])
        return task_id

    # ── 结果收集 ──

    def collect_notifications(self) -> list[str]:
        """收集已完成的后台任务，格式化为通知。

        每轮 agent_loop 调用一次。已完成的任务从内存中移除，
        （SQLite 中的记录保留用于审计）。

        Returns:
            <task_notification> 格式的通知文本列表
        """
        with self._lock:
            ready = [t for t in self._tasks.values()
                     if t.status in ("completed", "failed")]
            for t in ready:
                del self._tasks[t.task_id]

        notifications = []
        for task in ready:
            if task.status == "completed":
                summary = task.result[:300] + ("..." if len(task.result) > 300 else "")
                notifications.append(
                    f"<task_notification>\n"
                    f"  <task_id>{task.task_id}</task_id>\n"
                    f"  <status>completed</status>\n"
                    f"  <command>{task.command}</command>\n"
                    f"  <result>{summary}</result>\n"
                    f"</task_notification>"
                )
            else:
                notifications.append(
                    f"<task_notification>\n"
                    f"  <task_id>{task.task_id}</task_id>\n"
                    f"  <status>failed</status>\n"
                    f"  <command>{task.command}</command>\n"
                    f"  <error>{task.error}</error>\n"
                    f"</task_notification>"
                )
        return notifications

    # ── 优雅关闭 ──

    def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT):
        """优雅关闭：等运行中任务完成 → 存盘 → 关闭线程池。

        Args:
            timeout: 每个任务的最长等待秒数（默认 10s）
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        with self._lock:
            running = [t for t in self._tasks.values()
                       if t.status == "running"]
        if not running:
            self._close_executors()
            return

        print(f"\n⏳ 等待 {len(running)} 个后台任务完成（最多 {timeout:.0f}s）...")
        for task in running:
            print(f"  - {task.task_id}: {task.command[:60]}")

        deadline = time.time() + timeout
        interrupted_count = 0

        for task in running:
            remaining = deadline - time.time()
            if remaining <= 0:
                # 超时 → 中断
                self._interrupt_task(task)
                interrupted_count += 1
                continue

            try:
                if task._future:
                    task._future.result(timeout=remaining)
                    # worker 已完成，状态已由 worker 设为 completed/failed
            except Exception:
                # 超时或异常 → 中断
                self._interrupt_task(task)
                interrupted_count += 1

        self._close_executors()

        if interrupted_count:
            print(f"⚠ {interrupted_count} 个任务未完成，已保存状态，下次登录恢复。")

    def _interrupt_task(self, task: BackgroundTask):
        """中断未完成任务，保存状态到 SQLite。"""
        with self._lock:
            task.status = "interrupted"
            task.error = "进程关闭时中断"
            task.finished_at = datetime.now().isoformat()
        self._persist(task)
        logger.info("任务中断: %s", task.task_id)

    def _close_executors(self):
        """关闭所有线程池（不等运行中的任务——前面已经等过了）。"""
        for name, executor in self._executors.items():
            executor.shutdown(wait=False)
        self._executors.clear()

    # ── 持久化 ──

    def _persist(self, task: BackgroundTask):
        """更新 SQLite 中任务的状态。"""
        try:
            self._storage.update_background_task(
                task.task_id,
                status=task.status,
                result=task.result,
                error=task.error,
            )
        except Exception:
            logger.debug("后台任务状态持久化失败", exc_info=True)

    # ── 中断恢复 ──

    def resume_interrupted(self) -> list[dict]:
        """获取上次被中断的任务列表。

        登录时调用，告知用户有未完成的操作。
        """
        try:
            return self._storage.get_interrupted_tasks(self._username)
        except Exception:
            return []

    # ── 查询 ──

    @property
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values()
                       if t.status == "running")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    # ── 双重 Ctrl+C ──

    def on_interrupt(self) -> bool:
        """处理中断信号。

        Returns:
            True = 已经处理（应用继续运行）
            False = 应该立即退出
        """
        self._interrupt_count += 1
        if self._interrupt_count == 1:
            print("\n⚠ 收到中断信号，正在优雅关闭...（再按一次强制退出）")
            return True
        # 第二次 → 强杀
        print("\n⚠ 强制退出，未完成任务可能丢失。")
        return False

    def reset_interrupt(self):
        """登录时重置中断计数。"""
        self._interrupt_count = 0


# ═══════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════

_bg_manager: BackgroundTaskManager | None = None
_bg_lock = threading.Lock()


def get_bg_manager(storage=None, username: str = "") -> BackgroundTaskManager:
    """获取全局后台任务管理器。

    首次调用需提供 storage + username。
    """
    global _bg_manager
    with _bg_lock:
        if _bg_manager is None and storage is not None and username:
            _bg_manager = BackgroundTaskManager(storage, username)
        elif _bg_manager is None:
            raise RuntimeError(
                "BackgroundTaskManager 未初始化，请先调用 "
                "get_bg_manager(storage=..., username=...)"
            )
    return _bg_manager


def reset_bg_manager():
    """重置管理器（测试用 / 用户切换时）。"""
    global _bg_manager
    with _bg_lock:
        if _bg_manager is not None:
            _bg_manager.shutdown(timeout=0.5)
        _bg_manager = None


# ═══════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════

def _make_blocked_id(tool_name: str, reason: str) -> str:
    """生成表示拦截状态的伪 task_id。"""
    import uuid
    return f"blocked_{uuid.uuid4().hex[:8]}"
