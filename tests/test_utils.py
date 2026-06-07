"""测试: research_assistant.utils"""
import pytest


class TestJsonUtils:
    """P0-2: 统一 JSON 提取工具测试。"""

    def test_extract_plain_json(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        result = extract_json_from_llm_response('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_extract_markdown_json_block(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        content = '```json\n{"key": "value"}\n```'
        result = extract_json_from_llm_response(content)
        assert result == {"key": "value"}

    def test_extract_markdown_no_lang_block(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        content = '```\n{"key": "value"}\n```'
        result = extract_json_from_llm_response(content)
        assert result == {"key": "value"}

    def test_extract_nested_json(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        content = '{"papers": [{"title": "Test"}], "count": 5}'
        result = extract_json_from_llm_response(content)
        assert result == {"papers": [{"title": "Test"}], "count": 5}

    def test_extract_empty_raises(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        import json
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_llm_response("")

    def test_extract_invalid_raises(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        import json
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_llm_response("not json at all")

    def test_try_extract_json_returns_default(self):
        from research_assistant.utils.json_utils import try_extract_json
        result = try_extract_json("not json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_try_extract_json_success(self):
        from research_assistant.utils.json_utils import try_extract_json
        result = try_extract_json('{"ok": true}')
        assert result == {"ok": True}

    def test_extract_json_with_surrounding_text(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        content = 'Here is the result:\n```json\n{"score": 95}\n```\nHope this helps!'
        result = extract_json_from_llm_response(content)
        assert result == {"score": 95}

    def test_extract_json_list(self):
        from research_assistant.utils.json_utils import extract_json_from_llm_response
        content = '[{"name": "a"}, {"name": "b"}]'
        result = extract_json_from_llm_response(content)
        assert result == [{"name": "a"}, {"name": "b"}]


class TestLLMFactory:
    """P0-1: 统一 LLM 工厂测试。"""

    def test_get_llm_returns_chat_openai(self):
        from research_assistant.utils.llm_factory import get_llm
        llm = get_llm()
        from langchain_openai import ChatOpenAI
        assert isinstance(llm, ChatOpenAI)

    def test_get_llm_caches_same_params(self):
        from research_assistant.utils.llm_factory import get_llm, clear_llm_cache
        clear_llm_cache()
        llm1 = get_llm(temperature=0.3, max_tokens=2048)
        llm2 = get_llm(temperature=0.3, max_tokens=2048)
        assert llm1 is llm2

    def test_get_llm_different_params_different_instances(self):
        from research_assistant.utils.llm_factory import get_llm, clear_llm_cache
        clear_llm_cache()
        llm1 = get_llm(temperature=0.1)
        llm2 = get_llm(temperature=0.9)
        assert llm1 is not llm2

    def test_get_llm_no_cache(self):
        from research_assistant.utils.llm_factory import get_llm, clear_llm_cache
        clear_llm_cache()
        llm1 = get_llm(cache=False)
        llm2 = get_llm(cache=False)
        assert llm1 is not llm2

    def test_clear_llm_cache(self):
        from research_assistant.utils.llm_factory import get_llm, clear_llm_cache
        clear_llm_cache()
        llm1 = get_llm(temperature=0.5)
        clear_llm_cache()
        llm2 = get_llm(temperature=0.5)
        assert llm1 is not llm2

    def test_get_llm_custom_model(self):
        from research_assistant.utils.llm_factory import get_llm
        llm = get_llm(model="custom-model")
        assert llm.model_name == "custom-model"
