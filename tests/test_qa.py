"""QA Agent 服务测试 — 全 mock，无外部依赖。

测试新 Agent 架构:
  - LLM 自主工具调用（不再有固定意图路由）
  - 工具返回结果 → LLM 生成回答
  - 工作记忆累积
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import Mock, patch, MagicMock, ANY


class TestQAService:
    """QA Agent 服务核心链路测试。"""

    # ── helpers ──

    @staticmethod
    def _make_mock_llm(responses: list):
        """构造 mock LLM，invoke() 按顺序返回 responses。

        每个 response 是 AIMessage（可带 tool_calls）。
        """
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=responses)
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        return mock_llm

    @staticmethod
    def _make_ai_message(content: str, tool_calls: list = None):
        """构造模拟 AIMessage。

        Args:
            content: 回答文本
            tool_calls: None → 不设 tool_calls 属性（旧版兼容）
                         [] → 空列表（LLM 不调工具）
                         [...] → 工具调用列表
        """
        from langchain_core.messages import AIMessage
        msg = AIMessage(content=content)
        if tool_calls is not None:
            object.__setattr__(msg, 'tool_calls', tool_calls)
        return msg

    @staticmethod
    def _make_storage():
        """构造 mock PerUserStorage（含必需属性）。"""
        storage = MagicMock()
        storage.user_dir = "/tmp/test_user"
        storage.get_all_progress.return_value = []
        return storage

    # ── 核心测试 ──

    @patch("research_assistant.tools.qa.get_llm")
    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_chat_no_tools_called(self, mock_wm, mock_hr, mock_get_llm):
        """闲聊场景: LLM 不调工具，直接回答。"""
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        qa = QAService("t", self._make_storage())

        # LLM 直接回答，不调工具（空 tool_calls 列表）
        mock_llm = self._make_mock_llm([
            self._make_ai_message("你好！我是科研助手，有什么可以帮你？", tool_calls=[]),
        ])
        mock_get_llm.return_value = mock_llm

        result = qa.ask("你好")

        assert "你好" in result["answer"]
        assert result["tool_calls"] == []
        mock_hr.return_value.search_papers_expanded.assert_not_called()
        mock_wm.return_value.add_turn.assert_called_once()

    @patch("research_assistant.tools.qa.get_llm")
    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_agent_calls_tool_then_answers(self, mock_wm, mock_hr, mock_get_llm):
        """科研问题: Agent 调 query_knowledge_base，拿到结果后回答。"""
        from langchain_core.messages import AIMessage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[
            {
                "paper_id": 1, "title": "金刚石研磨工艺研究",
                "heading_path": "3.2 研磨参数",
                "abstract": "研究了不同粒度金刚石微粉的研磨效果。",
                "core_claim": "W10微粉在600rpm下表面粗糙度最优(Ra 2.1nm)。",
                "text": "实验表明W10微粉在600rpm转速下达到最低粗糙度Ra 2.1nm。",
                "payload": {"text": "W10微粉600rpm Ra 2.1nm"},
            },
        ])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        qa = QAService("t", self._make_storage())

        # 第1轮: LLM 决定调工具
        call1 = AIMessage(content="")
        object.__setattr__(call1, 'tool_calls', [{
            "name": "query_knowledge_base",
            "args": {"query": "金刚石研磨", "limit": 5},
            "id": "call_001",
        }])
        # 第2轮: LLM 拿到工具结果 → 生成回答
        call2 = self._make_ai_message(
            "根据论文[金刚石研磨工艺研究]，W10微粉在600rpm下粗糙度Ra 2.1nm为最优参数。",
            tool_calls=[],
        )
        mock_llm = self._make_mock_llm([call1, call2])
        mock_get_llm.return_value = mock_llm

        result = qa.ask("金刚石研磨的参数是多少")

        assert "W10" in result["answer"]
        assert "Ra 2.1" in result["answer"]
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "query_knowledge_base"
        mock_hr.return_value.search_papers_expanded.assert_called_once()
        mock_wm.return_value.add_turn.assert_called_once()

    @patch("research_assistant.tools.qa.get_llm")
    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_agent_calls_multiple_tools(self, mock_wm, mock_hr, mock_get_llm):
        """混合场景: Agent 同时调 query_knowledge_base + recall_history。"""
        from langchain_core.messages import AIMessage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[
            {
                "paper_id": 1, "title": "钙钛矿LED综述",
                "heading_path": "5. 光学性能",
                "core_claim": "PLQY最高达到95%。",
                "text": "钙钛矿量子点LED的PLQY最高达到95%，EQE超过20%。",
                "payload": {"text": "PLQY最高达到95%"},
            },
        ])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        qa = QAService("t", self._make_storage())

        # 第1轮: 同时调两个工具
        call1 = AIMessage(content="")
        object.__setattr__(call1, 'tool_calls', [
            {
                "name": "query_knowledge_base",
                "args": {"query": "钙钛矿 PLQY", "limit": 5},
                "id": "call_001",
            },
            {
                "name": "recall_history",
                "args": {"query": "钙钛矿 LED", "limit": 3},
                "id": "call_002",
            },
        ])
        call2 = self._make_ai_message(
            "根据论文[钙钛矿LED综述]，PLQY最高95%。历史讨论中也提到了相关进展。",
            tool_calls=[],
        )
        mock_llm = self._make_mock_llm([call1, call2])
        mock_get_llm.return_value = mock_llm

        result = qa.ask("钙钛矿LED的PLQY能做到多少，和上次讨论的对比一下")

        assert "PLQY" in result["answer"]
        assert "95%" in result["answer"]
        tool_names = {tc["tool"] for tc in result["tool_calls"]}
        assert "query_knowledge_base" in tool_names
        assert "recall_history" in tool_names

    @patch("research_assistant.tools.qa.get_llm")
    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_agent_no_results_honest_answer(self, mock_wm, mock_hr, mock_get_llm):
        """工具无结果时 Agent 诚实告知，不编造。"""
        from langchain_core.messages import AIMessage
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        qa = QAService("t", self._make_storage())

        # 第1轮: 调工具，第2轮: 诚实告知
        call1 = AIMessage(content="")
        object.__setattr__(call1, 'tool_calls', [{
            "name": "query_knowledge_base",
            "args": {"query": "火星探测器 抛光 参数", "limit": 5},
            "id": "call_003",
        }])
        call2 = self._make_ai_message(
            "抱歉，我在本地知识库中未找到关于该主题的相关论文。建议尝试外部搜索。",
            tool_calls=[],
        )
        mock_llm = self._make_mock_llm([call1, call2])
        mock_get_llm.return_value = mock_llm

        result = qa.ask("火星探测器的抛光参数")

        assert "未找到" in result["answer"] or "抱歉" in result["answer"]
        assert len(result["tool_calls"]) == 1

    # ── 工作记忆 ──

    @patch("research_assistant.tools.qa.get_llm")
    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_working_memory_accumulates(self, mock_wm, mock_hr, mock_get_llm):
        """工作记忆随对话累积。"""
        from research_assistant.tools.qa import QAService

        mock_hr.return_value.search_papers_expanded = Mock(return_value=[])
        mock_hr.return_value.search_progress = Mock(return_value=[])
        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.add_turn = Mock()
        mock_wm.return_value.stats.return_value = {"active_turns": 0}

        qa = QAService("t", self._make_storage())

        # 每次对话 LLM 直接回答
        msgs = [self._make_ai_message(f"回复{i}", tool_calls=[]) for i in range(3)]
        mock_llm = self._make_mock_llm(msgs)
        mock_get_llm.return_value = mock_llm

        for _ in range(3):
            qa.ask("你好")
        assert mock_wm.return_value.add_turn.call_count == 3

    # ── 会话管理 ──

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_end_session_clears_and_saves(self, mock_wm, mock_hr):
        """end_session 清理工作记忆并保存摘要。"""
        from research_assistant.tools.qa import QAService

        mock_wm.return_value.get_context = Mock(return_value="")
        mock_wm.return_value.get_session_summary = Mock(return_value={
            "topics": ["测试"], "key_discussions": ["讨论1"],
        })
        mock_wm.return_value.clear = Mock()

        qa = QAService("t", self._make_storage())
        qa.end_session()
        mock_wm.return_value.clear.assert_called_once()

    # ── _build_context ──

    @patch("research_assistant.tools.qa.HybridRetriever")
    @patch("research_assistant.tools.qa.WorkingMemory")
    def test_build_context_includes_profile_and_history(self, mock_wm, mock_hr):
        """_build_context 注入用户画像和工作记忆。"""
        from research_assistant.tools.qa import QAService

        mock_wm.return_value.get_context = Mock(
            return_value="Q: 上次问题\nA: 上次回答"
        )
        mock_wm.return_value.stats.return_value = {"active_turns": 1}

        storage = self._make_storage()
        storage.get_all_progress.return_value = [
            {"entry_type": "experiment", "title": "CVD测试",
             "content": "测试了600度", "insights": "600度比400度好"},
        ]
        qa = QAService("t", storage)

        ctx = qa._build_context()

        assert "上次问题" in ctx
        assert "CVD测试" in ctx
