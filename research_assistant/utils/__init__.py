"""科研助手工具集 — LLM 工厂、JSON 解析、文本处理等通用工具。"""
from .llm_factory import get_llm, clear_llm_cache
from .json_utils import extract_json_from_llm_response, try_extract_json
