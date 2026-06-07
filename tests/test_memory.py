"""测试: research_assistant.memory 核心函数"""
import pytest


class TestTokenizer:
    """retrieval.py _tokenize 测试。"""

    def test_tokenize_chinese(self):
        from research_assistant.rag.retrieval import _tokenize
        tokens = _tokenize("钙钛矿太阳能电池")
        # 中文按字切分
        assert "钙" in tokens
        assert "钛" in tokens
        assert "矿" in tokens

    def test_tokenize_english(self):
        from research_assistant.rag.retrieval import _tokenize
        tokens = _tokenize("perovskite solar cell")
        assert "perovskite" in tokens
        assert "solar" in tokens
        assert "cell" in tokens

    def test_tokenize_mixed(self):
        from research_assistant.rag.retrieval import _tokenize
        tokens = _tokenize("PLQY测试 perovskite stability")
        assert "plqy" in tokens
        assert "测" in tokens
        assert "perovskite" in tokens

    def test_tokenize_empty(self):
        from research_assistant.rag.retrieval import _tokenize
        tokens = _tokenize("")
        assert tokens == []


class TestRRFFusion:
    """retrieval.py RRF fusion 测试。"""

    def test_rrf_fuse_empty(self):
        from research_assistant.rag.retrieval import HybridRetriever
        r = HybridRetriever.__new__(HybridRetriever)
        result = r.rrf_fuse([], [])
        assert result == []

    def test_rrf_fuse_single_list(self):
        from research_assistant.rag.retrieval import HybridRetriever
        r = HybridRetriever.__new__(HybridRetriever)
        docs = [
            {"id": "a", "title": "Paper A"},
            {"id": "b", "title": "Paper B"},
        ]
        result = r.rrf_fuse([docs])
        assert len(result) == 2
        # a ranked higher because it appears first
        assert result[0]["id"] == "a"

    def test_rrf_fuse_two_lists_with_overlap(self):
        from research_assistant.rag.retrieval import HybridRetriever
        r = HybridRetriever.__new__(HybridRetriever)
        list1 = [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]
        list2 = [{"id": "b", "title": "B"}, {"id": "a", "title": "A"}]
        result = r.rrf_fuse([list1, list2])
        # Both a and b appear in both lists -> higher scores
        assert len(result) == 2
        # b gets higher RRF because ranked high in list2
        assert all("rrf_score" in doc for doc in result)

    def test_rrf_fuse_custom_key(self):
        from research_assistant.rag.retrieval import HybridRetriever
        r = HybridRetriever.__new__(HybridRetriever)
        docs = [{"doi": "10.1234", "title": "Test"}]
        result = r.rrf_fuse([docs], key_fn=lambda d: d.get("doi"))
        assert len(result) == 1
        assert result[0]["doi"] == "10.1234"


class TestSanitize:
    """user_manager.py _sanitize 测试。"""

    def test_sanitize_normal(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        assert um._sanitize("test_user") == "test_user"

    def test_sanitize_uppercase(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        assert um._sanitize("TestUser") == "testuser"

    def test_sanitize_special_chars(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        result = um._sanitize("user@name#123")
        assert "@" not in result
        assert "#" not in result

    def test_sanitize_chinese(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        result = um._sanitize("用户测试")
        assert all(c == '_' for c in result)  # Chinese chars replaced

    def test_sanitize_empty(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        assert um._sanitize("") == "default"

    def test_sanitize_too_long(self):
        from research_assistant.core.user_manager import UserManager
        um = UserManager(".")
        result = um._sanitize("a" * 50)
        assert len(result) == 30


class TestRateLimiter:
    """paper_search.py 线程安全速率限制器测试。"""

    def test_rate_limiter_creation(self):
        from research_assistant.tools.paper_search import _RateLimiter
        rl = _RateLimiter(min_interval=2.0, cooldown=60.0)
        assert rl._min_interval == 2.0
        assert rl._429_cooldown == 60.0

    def test_rate_limiter_initial_state(self):
        from research_assistant.tools.paper_search import _RateLimiter
        rl = _RateLimiter()
        assert rl._last_request == 0.0
        assert rl._429_until == 0.0

    def test_rate_limiter_mark_429(self):
        from research_assistant.tools.paper_search import _RateLimiter
        import time
        rl = _RateLimiter(cooldown=1.0)
        rl.mark_429()
        assert rl._429_until > time.time()
