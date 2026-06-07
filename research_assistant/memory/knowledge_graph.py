"""Neo4j 知识图谱存储 — 语义记忆的底层图数据库。

存储科研实体（材料/方法/性能/参数）及其关系，
支持跨论文推理、知识缺口检测、矛盾发现。

Schema:
  实体节点: Material, Method, Property, Parameter, Paper, Finding
  关系: HAS_PROPERTY, AFFECTS, USED_IN, CITED_BY, MEASURED_AS
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Neo4j 知识图谱管理器。"""

    # 允许的实体类型和关系类型（集中定义，防止 Cypher 注入）
    ALLOWED_ENTITY_TYPES = {"Material", "Method", "Property",
                            "Parameter", "Paper", "Finding"}
    ALLOWED_REL_TYPES = {"AFFECTS", "USED_IN", "HAS_PROPERTY",
                         "CITED_BY", "MEASURED_AS"}

    def __init__(self):
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            pwd = os.getenv("NEO4J_PASSWORD", "password")
            self._driver = GraphDatabase.driver(url, auth=(user, pwd))
        return self._driver

    def is_available(self) -> bool:
        """检查 Neo4j 是否可用。"""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── 初始化 Schema ──

    def init_schema(self):
        """创建约束和索引。"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Material) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Method) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Property) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Parameter) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Paper) REQUIRE n.paper_id IS UNIQUE",
        ]
        with self.driver.session() as s:
            for cypher in constraints:
                try:
                    s.run(cypher)
                except Exception:
                    logger.debug("约束创建失败（可能已存在）: %s", cypher[:50])

    # ── 实体 CRUD ──

    def upsert_entity(self, entity_type: str, name: str,
                       properties: dict | None = None):
        """创建或更新实体节点。

        Args:
            entity_type: Material | Method | Property | Parameter | Paper | Finding
            name: 实体名称（用于去重）
            properties: 额外属性
        """
        # 白名单校验 entity_type，防御 Cypher 注入
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            raise ValueError(f"非法的实体类型: {entity_type}")
        props = properties or {}
        props["name"] = name
        cypher = (
            f"MERGE (n:{entity_type} {{name: $name}}) "
            "SET n += $props"
        )
        with self.driver.session() as s:
            s.run(cypher, name=name, props=props)

    def upsert_relation(self, from_name: str, from_type: str,
                         to_name: str, to_type: str,
                         rel_type: str, properties: dict | None = None):
        """创建或更新两个实体之间的关系。

        Args:
            from_name: 源实体名称
            from_type: 源实体类型
            to_name: 目标实体名称
            to_type: 目标实体类型
            rel_type: 关系类型 (AFFECTS | USED_IN | HAS_PROPERTY | ...)
            properties: 关系属性
        """
        # 白名单校验
        if from_type not in self.ALLOWED_ENTITY_TYPES or to_type not in self.ALLOWED_ENTITY_TYPES:
            raise ValueError(f"非法的实体类型: {from_type} 或 {to_type}")
        if rel_type not in self.ALLOWED_REL_TYPES:
            raise ValueError(f"非法的关系类型: {rel_type}")
        props = properties or {}
        cypher = (
            f"MATCH (a:{from_type} {{name: $from_name}}) "
            f"MATCH (b:{to_type} {{name: $to_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += $props"
        )
        with self.driver.session() as s:
            s.run(cypher, from_name=from_name, to_name=to_name, props=props)

    def upsert_paper_entity(self, from_name: str, from_type: str,
                             to_name: str, to_type: str,
                             rel_type: str, paper_id: int):
        """创建实体关系并关联到论文。

        关系上标记 paper_id 以追溯来源。
        """
        props = {"paper_id": paper_id}
        self.upsert_relation(from_name, from_type, to_name, to_type,
                             rel_type, properties=props)

    # ── 图查询 ──

    def query_entity_neighbors(self, name: str, entity_type: str,
                                depth: int = 1) -> list[dict]:
        """查询一个实体的邻居（关联的实体和关系）。

        Returns:
            [{"entity": str, "type": str, "relation": str, "direction": "in"|"out"}, ...]
        """
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            raise ValueError(f"非法的实体类型: {entity_type}")
        cypher = (
            f"MATCH (n:{entity_type} {{name: $name}})-[r]-(m) "
            "RETURN m.name AS entity, labels(m)[0] AS type, "
            "type(r) AS relation, "
            "CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction "
            "LIMIT 50"
        )
        with self.driver.session() as s:
            result = s.run(cypher, name=name)
            return [dict(record) for record in result]

    def query_related_papers(self, name: str, entity_type: str) -> list[dict]:
        """查询与该实体相关的所有论文。"""
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            raise ValueError(f"非法的实体类型: {entity_type}")
        cypher = (
            f"MATCH (n:{entity_type} {{name: $name}})-[r]-(:Method|Material|Property|Parameter)"
            "WHERE r.paper_id IS NOT NULL "
            "RETURN DISTINCT r.paper_id AS paper_id, "
            "type(r) AS relation "
            "ORDER BY paper_id"
        )
        with self.driver.session() as s:
            result = s.run(cypher, name=name)
            return [dict(record) for record in result]

    def query_path(self, from_name: str, to_name: str,
                   max_depth: int = 3) -> list[dict]:
        """查询两个实体之间的路径。

        用于跨论文推理："金刚石薄膜" 和 "表面粗糙度" 之间有什么关联？
        """
        cypher = (
            "MATCH path = (a {name: $from_name})-[*1..%d]-(b {name: $to_name}) "
            "RETURN [n IN nodes(path) | n.name] AS entities, "
            "[r IN relationships(path) | type(r)] AS relations "
            "LIMIT 10"
        ) % max_depth
        with self.driver.session() as s:
            result = s.run(cypher, from_name=from_name, to_name=to_name)
            return [dict(record) for record in result]

    def get_statistics(self) -> dict:
        """获取图谱统计信息。"""
        stats = {}
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) AS cnt",
            "by_type": (
                "MATCH (n) "
                "RETURN labels(n)[0] AS type, count(n) AS cnt "
                "ORDER BY cnt DESC"
            ),
            "total_relations": "MATCH ()-[r]->() RETURN count(r) AS cnt",
        }
        with self.driver.session() as s:
            stats["total_nodes"] = s.run(queries["total_nodes"]).single()["cnt"]
            stats["total_relations"] = s.run(queries["total_relations"]).single()["cnt"]
            stats["by_type"] = [
                dict(r) for r in s.run(queries["by_type"])
            ]
        return stats

    def find_knowledge_gaps(self, entity_type: str = "Property") -> list[dict]:
        """发现知识缺口：某类实体中关联论文数量最少的。

        Returns:
            按关联论文数升序排列的实体列表（缺论文的排最前）
        """
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            raise ValueError(f"非法的实体类型: {entity_type}")
        cypher = (
            f"MATCH (n:{entity_type}) "
            "OPTIONAL MATCH (n)-[r]-(:Method|Material) "
            "WHERE r.paper_id IS NOT NULL "
            "WITH n, count(DISTINCT r.paper_id) AS paper_count "
            "RETURN n.name AS name, paper_count "
            "ORDER BY paper_count ASC "
            "LIMIT 20"
        )
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher)]

    def find_contradictions(self) -> list[dict]:
        """发现潜在矛盾：同一实体被多个论文关联到不同结论。"""
        cypher = (
            "MATCH (p1:Paper)-[r1]->(n)<-[r2]-(p2:Paper) "
            "WHERE p1.paper_id <> p2.paper_id AND type(r1) = type(r2) "
            "RETURN n.name AS entity, labels(n)[0] AS type, "
            "p1.paper_id AS paper1, p2.paper_id AS paper2, "
            "type(r1) AS shared_relation "
            "LIMIT 20"
        )
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher)]
