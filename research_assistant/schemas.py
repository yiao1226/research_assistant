"""LLM 结构化输出 Schema — Pydantic 模型约束 LLM JSON 输出。

面试可讲:
  替代 extract_json_from_llm_response() 的正则硬解析。
  llm.with_structured_output(Model) → LLM 直接返回 Pydantic 对象，
  格式错误自动 retry，保证下游代码拿到的数据一定合法。
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


# ── 搜索编排 ──

class ExpandedQuery(BaseModel):
    """扩展搜索词"""
    query: str = Field(description="扩展后的搜索词（英文，支持布尔）")
    rationale: str = Field(description="扩展理由")
    priority: Literal["high", "medium", "low"] = "medium"


class QueryExpansionResult(BaseModel):
    """LLM 关键词扩展输出"""
    expanded_queries: list[ExpandedQuery] = Field(min_length=1, max_length=5)
    search_focus: str = Field(description="搜索重点")
    exclude_directions: str = Field(default="")


class RankedPaper(BaseModel):
    """LLM 排序的单篇论文"""
    index: int = Field(ge=1, description="序号 (1-based)")
    core_contribution: str = Field(description="核心贡献（1句话）")
    innovation: str = Field(default="")
    relevance_reason: str = Field(description="与查询的相关性")
    ranking_reason: str = Field(default="")


class RankingResult(BaseModel):
    """LLM 排序输出"""
    ranked: list[RankedPaper] = Field(min_length=1)


# ── 入库标注 ──

class AnnotationMethod(BaseModel):
    """标注中的单个方法条目"""
    method_name: str
    method_category: Literal["fabrication", "characterization", "simulation", "analysis"] = "characterization"
    key_parameters: dict[str, str] = Field(default_factory=dict)
    what_it_measures: str = ""
    equipment: str = ""


class PaperAnnotation(BaseModel):
    """LLM 论文厚标注输出"""
    keywords_material: list[str] = Field(min_length=1, description="材料/化合物名称，中英双语")
    keywords_method: list[str] = Field(min_length=1, description="方法/技术/工艺名称")
    keywords_phenomenon: list[str] = Field(default_factory=list, description="现象/指标/性能名称")
    key_findings: list[str] = Field(min_length=1, max_length=8, description="定量或定性发现，保留数值")
    contribution_type: Literal["review", "experiment", "theory", "benchmark"] = "experiment"
    core_claim: str = Field(max_length=200, description="一句话核心结论")
    solved_problem: str = Field(default="")
    remaining_gap: str = Field(default="")
    baseline_methods: list[str] = Field(default_factory=list)
    improvement_over_baseline: str = Field(default="")
    methods: list[AnnotationMethod] = Field(default_factory=list, description="方法详情")
