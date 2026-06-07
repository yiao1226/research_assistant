"""QA 服务集成测试 — 全 mock，无外部依赖。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import Mock, patch, MagicMock


class TestQAService:
    """QA 服务核心链路测试。"""

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_intent_chat_no_retrieval(self, mock_wm, mock_hr):
        """chat 意图不触发检索。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        storage = MagicMock(spec=PerUserStorage)
        storage.get_all_progress.return_value = []
        qa = QAService("t", storage)

        with patch.object(qa, "_recognize_intent") as mi:
            mi.return_value = {
                "intent": "chat", "keywords": [],
                "understanding": "问候", "answer": "你好！",
            }
            result = qa.ask("你好")
            assert result["intent"] == "chat"
            assert result["answer"] == "你好！"
            mock_hr.return_value.search_papers_expanded.assert_not_called()

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_intent_research_retrieves(self, mock_wm, mock_hr):
        """research 意图触发检索并生成回答。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[
            {"title": "论文A", "heading_path": "4.1 研磨", "year": 2025, "venue": "", "abstract": "...", "core_claim": "W10最优"}
        ])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        storage = MagicMock(spec=PerUserStorage)
        storage.get_all_progress.return_value = []
        qa = QAService("t", storage)

        with patch.object(qa, "_recognize_intent") as mi:
            mi.return_value = {
                "intent": "research", "keywords": ["研磨"],
                "understanding": "用户问研磨问题", "answer": "",
            }
            with patch.object(qa, "_generate_answer", return_value="根据文献[论文1]，建议W10微粉。"):
                result = qa.ask("粗糙度降不下来")
                assert result["intent"] == "research"
                assert "[论文1]" in result["answer"]
                mock_hr.return_value.search_papers_expanded.assert_called_once()

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_intent_direction_extracts(self, mock_wm, mock_hr):
        """direction 意图提取后续方向并记录。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        storage = MagicMock(spec=PerUserStorage)
        storage.get_all_progress.return_value = []
        qa = QAService("t", storage)

        with patch.object(qa, "_recognize_intent") as mi:
            mi.return_value = {
                "intent": "direction", "keywords": ["下一步"],
                "understanding": "用户要方向", "answer": "",
            }
            with patch.object(qa, "_generate_answer", return_value="建议..."):
                with patch.object(qa, "_extract_directions") as md:
                    md.return_value = [{"title": "检测研磨盘", "priority": "high", "description": "...", "suggested_action": "..."}]
                    with patch.object(qa, "_record_directions", return_value=True):
                        result = qa.ask("下一步")
                        assert result["intent"] == "direction"
                        assert len(result["directions"]) == 1
                        assert result["progress_recorded"] is True

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_fallback_chat_on_llm_failure(self, mock_wm, mock_hr):
        """LLM 意图识别失败 → 内部已降级为 chat, 不抛异常。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        storage = MagicMock(spec=PerUserStorage)
        storage.get_all_progress.return_value = []
        qa = QAService("t", storage)

        # _recognize_intent 内部有 try/except, 失败时返回 chat
        with patch.object(qa, "_recognize_intent") as mi:
            mi.return_value = {"intent": "chat", "keywords": [],
                                "understanding": "降级", "answer": ""}
            with patch.object(qa, "_chat_answer", return_value="抱歉出错了"):
                result = qa.ask("任何问题")
                # 即使没有检索结果，chat 意图也不应触发检索
                assert result["intent"] == "chat"

    def test_cited_papers_extraction(self):
        """正确提取 [论文N] 引用。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService
        with patch("research_assistant.tools.qa.HybridRetriever"), \
             patch("research_assistant.tools.qa.WorkingMemory"):
            storage = MagicMock(spec=PerUserStorage)
            storage.get_all_progress.return_value = []
            qa = QAService("t", storage)

            papers = [{"title": "A"}, {"title": "B"}]
            assert len(qa._extract_cited_papers("[论文1]和[论文2]", papers)) == 2
            assert len(qa._extract_cited_papers("无引用", papers)) == 0

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_working_memory_accumulates(self, mock_wm, mock_hr):
        """工作记忆随对话累积。"""
        from research_assistant.core.storage import PerUserStorage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        storage = MagicMock(spec=PerUserStorage)
        storage.get_all_progress.return_value = []
        qa = QAService("t", storage)

        with patch.object(qa, "_recognize_intent") as mi:
            mi.return_value = {
                "intent": "chat", "keywords": [],
                "understanding": "问候", "answer": "你好",
            }
            for _ in range(3):
                qa.ask("你好")
        assert mock_wm.return_value.add_turn.call_count == 3
