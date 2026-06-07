"""语义记忆 — 基于 Neo4j 知识图谱的跨论文推理。

核心能力:
  1. LLM 实体-关系抽取: 从论文标注和 chunk heading_path 中抽取结构化知识
  2. 图谱写入: 实体/关系存入 Neo4j
  3. 图谱查询: 实体邻居、路径、相关论文
  4. 跨论文推理: 知识缺口检测、矛盾发现、研究方向建议
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils import get_llm, extract_json_from_llm_response
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """你是科研知识图谱构建专家。从论文标注和章节结构中提取实体和关系。

## 输入
论文标题: {title}
核心贡献: {core_claim}
贡献类型: {contribution_type}
关键词(材料): {keywords_material}
关键词(方法): {keywords_method}
关键词(现象/性能): {keywords_phenomenon}
关键发现: {key_findings}
章节结构: {heading_paths}

## 提取规则
1. 实体名称必须是概念性名称，不含具体数值。
   正确: "表面粗糙度", "载流子迁移率"
   错误: "表面粗糙度Ra 0.6μm", "迁移率180 cm²/V·s"
   具体数值可放在实体的 properties 字段中: {{"name": "表面粗糙度", "value": "Ra 0.6μm"}}
2. 关系方向要符合逻辑: Material-[HAS_PROPERTY]->Property, Method-[AFFECTS]->Property
3. 每个实体至少关联一个关系
4. 如果一个实体在多个章节被讨论，只提取一次
5. 参数类实体(Parameter)用范围/典型值: {{"name": "研磨压力", "value": "20N"}}

## 输出 JSON
{{
  "entities": [
    {{"name": "实体名称(概念)", "type": "Material|Method|Property|Parameter|Finding",
      "properties": {{"value": "具体数值(可选)"}}}}
  ],
  "relations": [
    {{
      "from_name": "源实体",
      "from_type": "源类型",
      "to_name": "目标实体",
      "to_type": "目标类型",
      "relation": "AFFECTS|USED_IN|HAS_PROPERTY|MEASURED_AS|PRODUCES"
    }}
  ]
}}

只返回 JSON。"""


class SemanticMemory:
    """语义记忆管理器 — 基于知识图谱的结构化知识存储和推理。"""

    def __init__(self, username: str):
        self.username = username
        self.kg = KnowledgeGraph()

    @property
    def available(self) -> bool:
        return self.kg.is_available()

    # ── 实体抽取与入库 ──

    def extract_and_store(self, paper_id: int, annotation: dict,
                           heading_paths: list[str] | None = None,
                           title: str = "") -> dict:
        """从论文标注中提取实体和关系，写入知识图谱。

        在 ingestion pipeline 中每入库一篇论文就调用一次。

        Args:
            paper_id: 论文 ID
            annotation: 厚标注 dict（关键词、方法、发现等）
            heading_paths: chunk 的标题路径列表
            title: 论文标题

        Returns:
            {"entity_count": N, "relation_count": M}
        """
        if not self.available:
            return {"entity_count": 0, "relation_count": 0,
                    "error": "Neo4j 不可用"}

        llm = get_llm(temperature=0.1, max_tokens=1024)

        # 构建 prompt 输入
        prompt_input = {
            "title": title or "未知",
            "core_claim": annotation.get("core_claim", ""),
            "contribution_type": annotation.get("contribution_type", ""),
            "keywords_material": json.dumps(
                annotation.get("keywords_material", []), ensure_ascii=False
            ),
            "keywords_method": json.dumps(
                annotation.get("keywords_method", []), ensure_ascii=False
            ),
            "keywords_phenomenon": json.dumps(
                annotation.get("keywords_phenomenon", []), ensure_ascii=False
            ),
            "key_findings": json.dumps(
                annotation.get("key_findings", []), ensure_ascii=False
            ),
            "heading_paths": ", ".join(heading_paths or []),
        }

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研知识图谱专家。只返回 JSON。"),
                HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(**prompt_input)),
            ])
            extracted = extract_json_from_llm_response(str(response.content))
        except Exception:
            logger.warning("论文 %d 实体抽取失败", paper_id)
            return {"entity_count": 0, "relation_count": 0, "error": "LLM 抽取失败"}

        entities = extracted.get("entities", [])
        relations = extracted.get("relations", [])

        # 写入图谱
        # Step 1: 论文节点
        self.kg.upsert_entity("Paper", str(paper_id), {
            "paper_id": paper_id,
            "name": title or f"Paper_{paper_id}",
        })

        # Step 2: 实体节点（携带数值属性）
        for ent in entities:
            self.kg.upsert_entity(
                ent["type"], ent["name"],
                properties=ent.get("properties") or {},
            )

        # Step 3: 关系 + 关联到论文
        for rel in relations:
            self.kg.upsert_paper_entity(
                from_name=rel["from_name"],
                from_type=rel["from_type"],
                to_name=rel["to_name"],
                to_type=rel["to_type"],
                rel_type=rel["relation"],
                paper_id=paper_id,
            )

        return {
            "entity_count": len(entities),
            "relation_count": len(relations),
        }

    # ── 图谱查询 ──

    def search_entity(self, name: str, entity_type: str = "") -> list[dict]:
        """查询实体的邻居和相关论文。"""
        if not self.available:
            return []
        types = [entity_type] if entity_type else [
            "Material", "Method", "Property", "Parameter"
        ]
        results = []
        for t in types:
            neighbors = self.kg.query_entity_neighbors(name, t)
            if neighbors:
                results.extend(neighbors)
                break
        return results

    def find_path(self, from_name: str, to_name: str) -> list[dict]:
        """查询两个实体之间的关联路径。"""
        if not self.available:
            return []
        return self.kg.query_path(from_name, to_name, max_depth=3)

    def get_knowledge_gaps(self) -> list[dict]:
        """获取知识缺口——关联论文最少的性能/属性。"""
        if not self.available:
            return []
        return self.kg.find_knowledge_gaps("Property")

    def get_contradictions(self) -> list[dict]:
        """获取潜在矛盾——同一实体被多篇论文关联。"""
        if not self.available:
            return []
        return self.kg.find_contradictions()

    def get_statistics(self) -> dict:
        """获取图谱统计。"""
        if not self.available:
            return {"error": "Neo4j 不可用"}
        return self.kg.get_statistics()

    # ── 跨论文推理 ──

    def suggest_research_direction(self, topic: str = "") -> str:
        """基于知识缺口和关系图谱，建议研究方向。

        Args:
            topic: 用户当前研究主题（可选，缩小建议范围）

        Returns:
            自然语言建议文本
        """
        if not self.available:
            return "语义记忆不可用（Neo4j 离线）"

        gaps = self.get_knowledge_gaps()
        stats = self.get_statistics()

        if not gaps or stats.get("total_nodes", 0) < 5:
            return "知识图谱数据不足，请先入库更多论文。"

        llm = get_llm(temperature=0.4, max_tokens=512)

        gaps_text = "\n".join(
            f"- {g['name']} (关联论文: {g['paper_count']}篇)"
            for g in gaps[:10]
        )
        stats_text = json.dumps(stats.get("by_type", []), ensure_ascii=False)

        prompt = f"""基于以下知识图谱信息，为科研人员建议 2-3 个有价值的研究方向。

当前研究主题: {topic or '未指定'}

知识缺口（关联论文最少的属性/性能）:
{gaps_text}

图谱统计:
{stats_text}

请给出具体的、可操作的建议，包含:
1. 为什么这个方向值得做
2. 可以从哪些已有知识出发
"""

        try:
            response = llm.invoke([
                SystemMessage(content="你是科研方向规划专家。"),
                HumanMessage(content=prompt),
            ])
            return str(response.content).strip()
        except Exception:
            return "建议生成失败，请稍后重试。"

    # ── 清理 ──

    def close(self):
        self.kg.close()
