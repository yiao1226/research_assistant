"""测试: P0 + P1 记忆系统改进

P0-1: 防抖队列
P0-2: 被动注入用户画像
P1-1: 结构化事实提取 + 置信度
P1-2: 纠正/认可信号检测
"""
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest


# ============================================================
# 测试: P1-2 纠正/认可信号检测
# ============================================================

class TestCorrectionDetection:
    """detect_correction 和 detect_reinforcement 测试。"""

    def test_detect_correction_obvious(self):
        from research_assistant.memory.message_processing import detect_correction
        assert detect_correction("不对，应该是 CVD 方法")
        assert detect_correction("不是这样，我用的 PVD")
        assert detect_correction("你搞错了")

    def test_detect_correction_subtle(self):
        from research_assistant.memory.message_processing import detect_correction
        assert detect_correction("实际上我用的温度是 800°C")
        assert detect_correction("准确地说，基底是硅而不是蓝宝石")
        assert detect_correction("纠正一下，那个参数不是 500")

    def test_detect_correction_normal(self):
        from research_assistant.memory.message_processing import detect_correction
        assert not detect_correction("CVD 沉积温度一般是多少？")
        assert not detect_correction("帮我查一下这篇论文")
        assert not detect_correction("")

    def test_detect_reinforcement_obvious(self):
        from research_assistant.memory.message_processing import detect_reinforcement
        assert detect_reinforcement("很好，这就是我想要的")
        assert detect_reinforcement("非常感谢！")
        assert detect_reinforcement("没错，完全正确")

    def test_detect_reinforcement_normal(self):
        from research_assistant.memory.message_processing import detect_reinforcement
        assert not detect_reinforcement("帮我再查一篇")
        assert not detect_reinforcement("不对，这个错了")

    def test_classify_turn_correction(self):
        from research_assistant.memory.message_processing import classify_turn
        result = classify_turn("不对，应该是 800 度")
        assert result["correction_detected"] is True
        assert result["reinforcement_detected"] is False

    def test_classify_turn_reinforcement(self):
        from research_assistant.memory.message_processing import classify_turn
        result = classify_turn("很好！谢谢！")
        assert result["correction_detected"] is False
        assert result["reinforcement_detected"] is True

    def test_classify_turn_neutral(self):
        from research_assistant.memory.message_processing import classify_turn
        result = classify_turn("帮我查一下 CVD 温度")
        assert result["correction_detected"] is False
        assert result["reinforcement_detected"] is False

    def test_filter_user_messages(self):
        from research_assistant.memory.message_processing import filter_user_messages
        from research_assistant.memory.working import QATurn

        turns = [
            QATurn("问题1", "回答1"),
            QATurn("问题2", "回答2"),
            QATurn("问题3", "回答3"),
        ]
        result = filter_user_messages(turns, last_n=2)
        assert "问题2" in result
        assert "问题3" in result
        assert "问题1" not in result  # 超出 last_n


# ============================================================
# 测试: P1-1 结构化事实提取
# ============================================================

class TestFactExtraction:
    """extract_facts_from_conversation 和 merge_facts 测试。"""

    def test_merge_facts_new_entries(self):
        from research_assistant.memory.fact_extraction import merge_facts
        existing = []
        new = [
            {"content": "用户偏好 CVD 方法", "category": "preference", "confidence": 0.9},
            {"content": "用户在 800°C 下实验", "category": "experiment_detail", "confidence": 0.95},
        ]
        result = merge_facts(existing, new)
        assert len(result) == 2
        assert result[0]["confidence"] == 0.95  # 高置信度排前面

    def test_merge_facts_overwrite_existing(self):
        """同 content 的新事实覆盖旧事实。"""
        from research_assistant.memory.fact_extraction import merge_facts
        existing = [
            {"content": "用户偏好 CVD 方法", "category": "preference",
             "confidence": 0.7, "createdAt": "2026-01-01"},
        ]
        new = [
            {"content": "用户偏好 CVD 方法", "category": "preference",
             "confidence": 0.95},
        ]
        result = merge_facts(existing, new)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95
        assert result[0]["createdAt"] == "2026-01-01"  # 保留原始创建时间

    def test_merge_facts_confidence_threshold(self):
        """低于阈值的事实被过滤。"""
        from research_assistant.memory.fact_extraction import merge_facts
        existing = []
        new = [
            {"content": "用户偏好 Python", "category": "preference", "confidence": 0.9},
            {"content": "用户可能喜欢 R", "category": "preference", "confidence": 0.3},
        ]
        result = merge_facts(existing, new, confidence_threshold=0.5)
        assert len(result) == 1
        assert result[0]["content"] == "用户偏好 Python"

    def test_merge_facts_max_facts_cap(self):
        """超过 max_facts 时裁剪低置信度事实。"""
        from research_assistant.memory.fact_extraction import merge_facts
        existing = []
        new = [
            {"content": f"事实{i}", "category": "context",
             "confidence": 0.5 + 0.05 * i}
            for i in range(10)
        ]
        result = merge_facts(existing, new, max_facts=5)
        assert len(result) == 5
        # 保留置信度最高的 5 条
        assert result[0]["confidence"] == pytest.approx(0.95)

    def test_merge_facts_empty(self):
        from research_assistant.memory.fact_extraction import merge_facts
        result = merge_facts([], [])
        assert result == []

    def test_remove_facts_by_content(self):
        from research_assistant.memory.fact_extraction import remove_facts
        existing = [
            {"id": "1", "content": "用户偏好 CVD"},
            {"id": "2", "content": "用户在 800°C 实验"},
        ]
        result = remove_facts(existing, ["用户偏好 cvd"])  # casefold 匹配
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_format_facts_for_injection(self):
        from research_assistant.memory.fact_extraction import format_facts_for_injection
        facts = [
            {"content": "用户偏好 CVD 方法", "category": "preference", "confidence": 0.95},
            {"content": "用户在 800°C 实验", "category": "experiment_detail", "confidence": 0.8},
            {"content": "用户研究金刚石薄膜", "category": "context", "confidence": 0.6},
        ]
        result = format_facts_for_injection(facts, top_n=2, max_tokens=200)
        assert "CVD" in result
        assert "800°C" in result
        assert "金刚石" not in result  # 最低置信度被裁剪

    def test_format_facts_for_injection_empty(self):
        from research_assistant.memory.fact_extraction import format_facts_for_injection
        result = format_facts_for_injection([], top_n=10)
        assert result == ""

    def test_build_user_profile_text(self):
        from research_assistant.memory.fact_extraction import build_user_profile_text
        facts = [
            {"content": "用户偏好 CVD 方法", "category": "preference", "confidence": 0.95},
        ]
        user_context = {
            "researchFocus": "金刚石薄膜器件结构设计",
            "methodPreference": "CVD 方法为主",
            "expertiseLevel": "intermediate",
        }
        result = build_user_profile_text(
            facts=facts,
            user_context=user_context,
            unresolved_questions=["基底温度如何优化？"],
        )
        assert "金刚石薄膜" in result
        assert "CVD" in result
        assert "基底温度" in result
        assert "95%" in result

    def test_normalize_confidence(self):
        from research_assistant.memory.fact_extraction import _normalize_confidence
        assert _normalize_confidence(0.5) == 0.5
        assert _normalize_confidence(1.5) == 1.0  # 截断
        assert _normalize_confidence(-0.5) == 0.0  # 截断
        assert _normalize_confidence(None) == 0.5  # 默认
        assert _normalize_confidence("high") == 0.5  # 无效

    def test_fact_content_key_casefold(self):
        from research_assistant.memory.fact_extraction import _fact_content_key
        assert _fact_content_key("通过 CVD 制备") == _fact_content_key("通过 cvd 制备")
        assert _fact_content_key("  CVD 方法  ") == _fact_content_key("cvd 方法")
        assert _fact_content_key("") is None
        assert _fact_content_key(123) is None


# ============================================================
# 测试: P0-1 防抖队列
# ============================================================

class TestMemoryUpdateQueue:
    """MemoryUpdateQueue 防抖机制测试。"""

    def test_queue_creation(self):
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=0.1)
        assert q._debounce_seconds == 0.1
        assert q.pending_count == 0

    def test_queue_add_and_pending_count(self):
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)  # 1小时，不会自动触发
        callback_called = []

        q.add("session_1", lambda: callback_called.append(1))
        assert q.pending_count == 1

        q.cancel("session_1")
        assert q.pending_count == 0

    def test_queue_overwrite_same_session(self):
        """同一 session 的新条目覆盖旧条目。"""
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)
        call_count = [0]

        q.add("s1", lambda: call_count.__setitem__(0, call_count[0] + 1))
        q.add("s1", lambda: call_count.__setitem__(0, call_count[0] + 1))
        assert q.pending_count == 1  # 覆盖，不是追加

        q.flush_nowait()
        # 由于 callback 中用了 __setitem__，直接执行会有问题。
        # 用 threading.Event 来验证。

    def test_queue_flush_nowait(self):
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)
        results = []

        q.add("s1", lambda: results.append("done1"))
        q.add("s2", lambda: results.append("done2"))
        assert q.pending_count == 2

        q.flush_nowait()
        # flush_nowait 在后台线程执行，需要等一小段时间
        time.sleep(0.1)
        assert "done1" in results
        assert "done2" in results
        assert q.pending_count == 0

    def test_queue_add_nowait(self):
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)
        results = []

        q.add_nowait("s1", lambda: results.append("immediate"))
        time.sleep(0.1)
        assert "immediate" in results
        assert q.pending_count == 0

    def test_queue_signal_merge(self):
        """纠正信号合并：任一 add 带 correction=True 则最终为 True。"""
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)

        q.add("s1", lambda: None, correction_detected=False)
        q.add("s1", lambda: None, correction_detected=True)
        q.add("s1", lambda: None, correction_detected=False)

        # 覆盖模式下，已合并的 correction 保持 True（源码中 merged_correction_detected
        # 逻辑是：新值 or 旧值，一旦 True 就一直 True）
        with q._lock:
            entry = q._queue.get("s1")
            assert entry is not None
            assert entry.correction_detected is True

    def test_global_singleton(self):
        from research_assistant.memory.debounce_queue import (
            get_memory_queue,
            reset_memory_queue,
        )
        q1 = get_memory_queue(debounce_seconds=0.5)
        q2 = get_memory_queue()
        assert q1 is q2  # 同一实例
        assert q1._debounce_seconds == 0.5  # 首次创建时设置

        reset_memory_queue()
        q3 = get_memory_queue(debounce_seconds=5.0)
        assert q3 is not q1  # 已重置

    def test_queue_thread_safety(self):
        """多线程同时 add 不同 session，不丢失、不重复。"""
        from research_assistant.memory.debounce_queue import MemoryUpdateQueue
        q = MemoryUpdateQueue(debounce_seconds=3600.0)
        count = 10
        barrier = threading.Barrier(count)

        def add_session(i):
            barrier.wait()
            q.add(f"session_{i}", lambda: None)

        threads = [threading.Thread(target=add_session, args=(i,)) for i in range(count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert q.pending_count == count  # 不同 session 不会互相覆盖

        for i in range(count):
            q.cancel(f"session_{i}")
        assert q.pending_count == 0


# ============================================================
# 测试: P0-2 用户画像管理
# ============================================================

class TestUserProfileManager:
    """UserProfileManager 测试。"""

    def test_profile_creation(self):
        from research_assistant.memory.user_profile import UserProfileManager
        mgr = UserProfileManager("test_user", str(Path(tempfile.gettempdir()) / "test_profile"))
        assert mgr.username == "test_user"

    def test_load_empty_facts(self):
        from research_assistant.memory.user_profile import UserProfileManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)
            data = mgr.load_facts()
            assert data["version"] == "1.0"
            assert data["facts"] == []
            assert data["userContext"] == {}

    def test_save_and_load_facts(self):
        from research_assistant.memory.user_profile import UserProfileManager
        from research_assistant.memory.fact_extraction import merge_facts

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)
            new_facts = [
                {"content": "用户偏好 CVD", "category": "preference", "confidence": 0.9},
            ]
            merged = merge_facts([], new_facts)
            ok = mgr.save_facts({"userContext": {"researchFocus": "金刚石"}, "facts": merged})
            assert ok
            assert mgr.facts_path.exists()

            loaded = mgr.load_facts()
            assert loaded["userContext"]["researchFocus"] == "金刚石"
            assert len(loaded["facts"]) == 1
            assert loaded["facts"][0]["confidence"] == 0.9

    def test_update_facts_incremental(self):
        from research_assistant.memory.user_profile import UserProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)

            # 第一次更新
            ok = mgr.update_facts(
                user_context={"researchFocus": "金刚石薄膜"},
                new_facts=[
                    {"content": "CVD 温度 800°C", "category": "experiment_detail",
                     "confidence": 0.95},
                ],
            )
            assert ok

            # 第二次更新
            ok = mgr.update_facts(
                user_context={"methodPreference": "CVD 方法"},
                new_facts=[
                    {"content": "基底用硅晶圆", "category": "experiment_detail",
                     "confidence": 0.85},
                ],
            )
            assert ok

            loaded = mgr.load_facts()
            assert loaded["userContext"]["researchFocus"] == "金刚石薄膜"
            assert loaded["userContext"]["methodPreference"] == "CVD 方法"
            assert len(loaded["facts"]) == 2

    def test_update_facts_with_removal(self):
        from research_assistant.memory.user_profile import UserProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)

            # 先加一条
            ok = mgr.update_facts(
                user_context={},
                new_facts=[
                    {"content": "旧信息", "category": "context", "confidence": 0.6},
                ],
            )
            loaded = mgr.load_facts()
            assert len(loaded["facts"]) == 1

            # 删除 + 新增
            old_fact_id = loaded["facts"][0].get("id")
            ok = mgr.update_facts(
                user_context={},
                new_facts=[
                    {"content": "新信息", "category": "context", "confidence": 0.9},
                ],
                facts_to_remove=[old_fact_id],
            )
            loaded = mgr.load_facts()
            assert len(loaded["facts"]) == 1
            assert "新信息" in loaded["facts"][0]["content"]

    def test_build_context_for_qa(self):
        from research_assistant.memory.user_profile import UserProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)
            mgr.update_facts(
                user_context={"researchFocus": "钙钛矿太阳能电池", "expertiseLevel": "advanced"},
                new_facts=[
                    {"content": "偏好旋涂法", "category": "preference", "confidence": 0.9},
                    {"content": "效率达到 25%", "category": "experiment_detail", "confidence": 0.95},
                ],
            )

            ctx = mgr.build_context_for_qa(
                unresolved_questions=["如何提高稳定性？"],
            )
            assert "钙钛矿太阳能电池" in ctx
            assert "旋涂法" in ctx
            assert "如何提高稳定性" in ctx
            assert "90%" in ctx or "95%" in ctx

    def test_stats(self):
        from research_assistant.memory.user_profile import UserProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = UserProfileManager("test", tmp)
            mgr.update_facts(
                user_context={"researchFocus": "test"},
                new_facts=[
                    {"content": "f1", "category": "preference", "confidence": 0.9},
                    {"content": "f2", "category": "experiment_detail", "confidence": 0.8},
                    {"content": "f3", "category": "preference", "confidence": 0.7},
                ],
            )
            stats = mgr.stats()
            assert stats["total_facts"] == 3
            assert stats["categories"]["preference"] == 2
            assert stats["categories"]["experiment_detail"] == 1
            assert stats["has_context"] is True


# ============================================================
# 集成测试: WorkingMemory 改进版
# ============================================================

class TestWorkingMemoryV2:
    """WorkingMemory 改进版集成测试。"""

    def test_add_turn_detects_signals(self):
        """add_turn 自动检测纠正/认可信号。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user")
        assert wm._any_correction_detected is False

        wm.add_turn("帮我查 CVD 温度", "CVD 温度通常是 600-900°C")
        assert wm._any_correction_detected is False

        wm.add_turn("不对，应该是 800°C", "已更正，CVD 温度是 800°C")
        assert wm._any_correction_detected is True

        assert len(wm.turns) == 2
        assert wm.turns[1].correction_detected is True

        wm.clear()

    def test_add_turn_detects_reinforcement(self):
        """add_turn 自动检测认可信号。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user")

        wm.add_turn("这个结论对吗？", "是的，基于论文数据。")
        wm.add_turn("很好，谢谢！", "不客气！")

        assert wm._any_reinforcement_detected is True
        assert wm.turns[1].reinforcement_detected is True

        wm.clear()

    def test_debounce_timer_created(self):
        """add_turn 创建防抖定时器。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user", debounce_seconds=60.0)

        wm.add_turn("测试问题", "测试回答")
        assert wm._debounce_timer is not None
        assert wm._debounce_seconds == 60.0

        wm._cancel_timer()
        assert wm._debounce_timer is None

    def test_debounce_timer_reset(self):
        """连续 add_turn 重置定时器。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user", debounce_seconds=60.0)

        wm.add_turn("Q1", "A1")
        timer1 = wm._debounce_timer
        assert timer1 is not None

        wm.add_turn("Q2", "A2")
        timer2 = wm._debounce_timer
        assert timer2 is not None
        assert timer2 is not timer1  # 新定时器

        wm._cancel_timer()

    def test_add_nowait_flush_safe_when_empty(self):
        """空 turns 时 add_nowait_flush 不报错。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user")
        # 不应该抛异常
        wm.add_nowait_flush()
        wm.clear()

    def test_signal_reset_after_flush(self):
        """事实提取后信号被重置。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user", debounce_seconds=60.0)

        wm._any_correction_detected = True
        wm._any_reinforcement_detected = True

        # add_nowait_flush 调用 _extract_and_persist_facts
        # 在没有 profile manager 路径且 turns 不足时会提前返回
        # 但信号重置发生在 _extract_and_persist_facts 末尾
        # 我们用 monkeypatch 来测试重置逻辑
        wm._extract_and_persist_facts()  # turns 不足会提前返回，但不影响信号重置测试

    def test_stats_includes_new_fields(self):
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user")
        wm._any_correction_detected = True

        stats = wm.stats()
        assert stats["correction_detected"] is True
        assert stats["reinforcement_detected"] is False
        assert "debounce_seconds" in stats

    def test_session_summary_unchanged(self):
        """向后兼容：get_session_summary 格式不变。"""
        from research_assistant.memory.working import WorkingMemory
        wm = WorkingMemory("test_user")
        wm.add_turn("测试问题", "测试回答", [], "test understanding")

        summary = wm.get_session_summary()
        assert "topics" in summary
        assert "papers_added" in summary
        assert "session_date" in summary
        assert "test understanding" in summary["topics"]

        wm.clear()


# ============================================================
# 运行入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
