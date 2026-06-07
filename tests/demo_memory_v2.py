"""记忆系统改进 — 直白演示脚本

每个改进独立演示，打印输入→输出，一眼看出改了什么。

用法: python tests/demo_memory_v2.py
"""
import sys
import time
import threading
from pathlib import Path

# 确保项目根目录在 Python 路径中（无论从哪里执行都能找到 research_assistant）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------- 演示 1: 纠正信号检测 ----------

def demo_1_signal_detection():
    """P1-2: 用户说"不对"时，系统能自动检测到。"""
    print("=" * 60)
    print("演示 1: 纠正/认可信号检测")
    print("=" * 60)

    from research_assistant.memory.message_processing import detect_correction, detect_reinforcement

    test_cases = [
        ("不对，应该是 CVD 方法", "纠正"),
        ("你搞错了，我用的温度是 800°C", "纠正"),
        ("帮我查一下这篇论文", "正常"),
        ("很好！谢谢你", "认可"),
        ("没错，完全正确", "认可"),
        ("", "边界: 空字符串"),
    ]

    for text, label in test_cases:
        corr = "[纠正]" if detect_correction(text) else ""
        rein = "[认可]" if detect_reinforcement(text) else ""
        flag = (corr + rein) if (corr or rein) else "[正常]"
        print(f"  [{flag}] {label}: \"{text}\"")
    print()


# ---------- 演示 2: 防抖队列 ----------

def demo_2_debounce_queue():
    """P0-1: 30秒防抖，新消息覆盖旧消息，退出时紧急刷盘。"""
    print("=" * 60)
    print("演示 2: 防抖记忆更新队列")
    print("=" * 60)

    from research_assistant.memory.debounce_queue import MemoryUpdateQueue

    # 设 0.5 秒防抖（演示用，生产环境 30 秒）
    q = MemoryUpdateQueue(debounce_seconds=0.5)
    log = []

    print("  用户: '我叫张三'")
    q.add("user_A", lambda: log.append("张三的信息已写入记忆"))
    print(f"  → 队列中有 {q.pending_count} 条待处理")

    print("  用户: '我是后端工程师'（3秒后）")
    q.add("user_A", lambda: log.append("张三（后端工程师）的信息已写入记忆"))
    print(f"  → 队列中仍为 {q.pending_count} 条（覆盖，不是追加）")

    print("  等待 0.6 秒让定时器触发...")
    time.sleep(0.6)
    print(f"  → 定时器已触发！log = {log}")
    print(f"  → 队列中剩余 {q.pending_count} 条")

    # 演示紧急刷盘
    print("\n  演示紧急刷盘:")
    q.add("user_B", lambda: log.append("李四的信息已写入记忆"))
    print(f"  → 队列中有 {q.pending_count} 条（李四的）")
    q.flush_nowait()
    time.sleep(0.1)
    print(f"  → flush_nowait 后 log = {log}")
    print(f"  → 队列中剩余 {q.pending_count} 条")
    print()


# ---------- 演示 3: 事实合并 ----------

def demo_3_fact_merging():
    """P1-1: 新旧事实按置信度合并，低置信度过滤。"""
    print("=" * 60)
    print("演示 3: 结构化事实提取与合并")
    print("=" * 60)

    from research_assistant.memory.fact_extraction import merge_facts

    # 模拟已有事实
    old_facts = [
        {"content": "用户偏好 Python", "category": "preference", "confidence": 0.7},
    ]
    print(f"  已有事实: {len(old_facts)} 条")
    for f in old_facts:
        print(f"    - [{f['category']}] {f['content']} (置信度:{f['confidence']})")

    # 新提取的事实
    new_facts = [
        {"content": "用户偏好 Python", "category": "preference",     "confidence": 0.95},  # 覆盖
        {"content": "用户在做 CVD 实验", "category": "experiment_detail", "confidence": 0.9},
        {"content": "用户可能也用过 R", "category": "preference",  "confidence": 0.3},   # 丢弃
        {"content": "基底温度 800°C", "category": "experiment_detail", "confidence": 0.85},
    ]
    print(f"\n  新提取: {len(new_facts)} 条")
    for f in new_facts:
        print(f"    - [{f['category']}] {f['content']} (置信度:{f['confidence']})")

    result = merge_facts(old_facts, new_facts, confidence_threshold=0.5, max_facts=100)

    print(f"\n  合并结果: {len(result)} 条（按置信度排序）")
    for f in result:
        status = "[+]" if f["confidence"] >= 0.9 else "  "
        print(f"    {status} [{f['category']}] {f['content']} (置信度:{f['confidence']})")

    print(f"\n  说明: '偏好 Python' 置信度从 0.7 → 0.95（覆盖）")
    print(f"         '可能用过 R' 置信度 0.3 < 阈值 0.5（丢弃）")
    print()


# ---------- 演示 4: 用户画像 ----------

def demo_4_user_profile():
    """P0-2: 用户画像存储和注入。"""
    print("=" * 60)
    print("演示 4: 用户画像被动注入")
    print("=" * 60)

    import tempfile
    from research_assistant.memory.user_profile import UserProfileManager
    from research_assistant.memory.fact_extraction import build_user_profile_text

    with tempfile.TemporaryDirectory() as tmp:
        mgr = UserProfileManager("张三", tmp)

        # 模拟多次对话后积累的事实
        mgr.update_facts(
            user_context={
                "researchFocus": "金刚石薄膜 CVD 沉积工艺优化",
                "methodPreference": "以微波等离子体 CVD 为主",
                "expertiseLevel": "advanced",
            },
            new_facts=[
                {"content": "CVD 沉积温度 800-900°C", "category": "experiment_detail", "confidence": 0.95},
                {"content": "基底材料偏好单晶硅(100)", "category": "experiment_detail", "confidence": 0.9},
                {"content": "偏好中文综述和英文研究论文", "category": "preference", "confidence": 0.85},
                {"content": "关注金刚石薄膜在量子传感中的应用", "category": "context", "confidence": 0.8},
            ],
        )

        profile_text = mgr.build_context_for_qa(
            unresolved_questions=["如何在保持薄膜质量的同时提高生长速率？"]
        )

        print("  下次对话时，以下内容会被自动注入到 Agent 的上下文中:\n")
        print(f"  {profile_text.replace(chr(10), chr(10) + '  ')}")
        print()

        stats = mgr.stats()
        print(f"  画像统计: {stats['total_facts']} 条事实, "
              f"分类: {stats['categories']}, "
              f"最后更新: {stats['last_updated'][:19]}")
    print()


# ---------- 演示 5: 工作记忆改进版 ----------

def demo_5_working_memory():
    """WorkingMemory 改进版：信号检测+防抖触发器。"""
    print("=" * 60)
    print("演示 5: 工作记忆改进版（信号检测 + 防抖）")
    print("=" * 60)

    from research_assistant.memory.working import WorkingMemory
    wm = WorkingMemory("测试用户", debounce_seconds=60.0)  # 60秒，演示时不触发

    print("  模拟对话:")
    wm.add_turn("帮我查 CVD 温度", "CVD 温度通常是 600-900°C")
    print("    第1轮: 问='帮我查 CVD 温度'")
    print(f"      信号: 纠正={wm.turns[-1].correction_detected}, "
          f"认可={wm.turns[-1].reinforcement_detected}")

    wm.add_turn("不对，应该是 800°C，我实验用的是 800", "已更正，CVD 温度是 800°C")
    print("    第2轮: 问='不对，应该是 800°C，我实验用的是 800'")
    print(f"      信号: 纠正={wm.turns[-1].correction_detected}, "
          f"认可={wm.turns[-1].reinforcement_detected}")

    wm.add_turn("很好，谢谢！", "不客气！")
    print("    第3轮: 问='很好，谢谢！'")
    print(f"      信号: 纠正={wm.turns[-1].correction_detected}, "
          f"认可={wm.turns[-1].reinforcement_detected}")

    stats = wm.stats()
    print(f"\n  工作记忆状态: {stats}")
    print(f"  防抖定时器已启动: {wm._debounce_timer is not None}")

    wm._cancel_timer()
    wm.clear()
    print()


# ---------- 主程序 ----------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  科研助手 -- 记忆系统 P0+P1 改进直白演示")
    print("=" * 60)
    print()

    demo_1_signal_detection()
    demo_2_debounce_queue()
    demo_3_fact_merging()
    demo_4_user_profile()
    demo_5_working_memory()

    print("[OK] 全部演示完毕")
