"""评估查询集 — 基于 eval 用户知识库的 5 篇真实论文。

论文:
  [1] 大尺寸金刚石薄膜的制备与精密研磨抛光工艺研究
  [2] 基于钙钛矿单晶的低噪型X射线探测平...
  [3] 基于钙钛矿单晶的高性能X射线/γ射线探测...
  [4] MAPbBr3单晶薄膜制备及其偏振探测
  [5] Six-Inch High-Purity Lead Halide Perovskite Wafer...

查询设计覆盖的检索挑战:
  - 跨语言: 中文 query ↔ 英文论文 (paper 5)
  - 同义词: "钙钛矿" ↔ "perovskite" ↔ "MAPbBr3" ↔ "lead halide"
  - 精确术语: "金刚石薄膜" vs "钙钛矿薄膜" → 不同论文
  - 多匹配: 一个 query 对应多篇相关论文
  - 语义泛化: "光电材料" → 应匹配但不直接包含该词
"""

from __future__ import annotations

# (query_text, relevant_paper_ids, difficulty, category, notes)
EVAL_QUERIES: list[dict] = [
    # ═══ Easy: 直接关键词匹配 ═══
    {
        "id": "q1",
        "query": "金刚石薄膜的研磨抛光工艺",
        "relevant_ids": {1},
        "difficulty": "easy",
        "category": "direct_keyword",
        "notes": "论文1的核心主题，关键词高度匹配",
    },
    {
        "id": "q2",
        "query": "MAPbBr3 单晶薄膜制备",
        "relevant_ids": {4},
        "difficulty": "easy",
        "category": "direct_keyword",
        "notes": "论文4的精确主题词，MAPbBr3 是强区分词",
    },

    # ═══ Medium: 同义词/跨语言 ═══
    {
        "id": "q3",
        "query": "钙钛矿单晶 X 射线探测器",
        "relevant_ids": {2, 3},
        "difficulty": "medium",
        "category": "multi_match",
        "notes": "论文2和3都做钙钛矿单晶X射线探测，应同时召回",
    },
    {
        "id": "q4",
        "query": "perovskite wafer fabrication",
        "relevant_ids": {5},
        "difficulty": "medium",
        "category": "cross_language",
        "notes": "英文 query 搜唯一英文论文，但中文论文也含 perovskite 关键词",
    },
    {
        "id": "q5",
        "query": "金刚石表面粗糙度优化",
        "relevant_ids": {1},
        "difficulty": "medium",
        "category": "synonym",
        "notes": "论文1含'表面粗糙度'标注但不一定在标题中直接出现",
    },

    # ═══ Hard: 语义理解/跨领域 ═══
    {
        "id": "q6",
        "query": "卤化铅钙钛矿的大面积制备方法",
        "relevant_ids": {5},
        "difficulty": "hard",
        "category": "semantic",
        "notes": "论文5是6英寸晶圆，但需理解 '卤化铅钙钛矿'='lead halide perovskite'",
    },
    {
        "id": "q7",
        "query": "钙钛矿材料的缺陷调控与电学性能",
        "relevant_ids": {2, 3, 4},
        "difficulty": "hard",
        "category": "multi_match_semantic",
        "notes": "三篇都涉及缺陷/迁移率/电学性能，需语义理解而非精确匹配",
    },
    {
        "id": "q8",
        "query": "单晶生长工艺优化",
        "relevant_ids": {3, 4},
        "difficulty": "hard",
        "category": "implicit",
        "notes": "论文3变组成生长+论文4空间限域法→都是单晶工艺优化，但query用词不精确",
    },

    # ═══ Edge case: 故意排除 ═══
    {
        "id": "q9",
        "query": "CVD 金刚石薄膜沉积",
        "relevant_ids": set(),  # 库里无 CVD 沉积论文！paper1 是研磨抛光，非沉积
        "difficulty": "trap",
        "category": "negative",
        "notes": "关键测试: 检索系统应返回空或低相关，不应错误匹配 paper1（它涉及金刚石但不涉及CVD沉积）",
    },
    {
        "id": "q10",
        "query": "钙钛矿晶圆的透明度和光学性能",
        "relevant_ids": {5},
        "difficulty": "medium",
        "category": "semantic",
        "notes": "论文5标注含'高光学透明度'，需语义关联'透明'→'transparency'",
    },
]

# 消融实验的 4 组检索变体
VARIANTS = {
    "bm25_only": {
        "label": "BM25-only (关键词)",
        "desc": "纯 jieba 分词 + BM25Okapi，无向量",
    },
    "dense_only": {
        "label": "Dense-only (语义向量)",
        "desc": "纯 BGE-small-zh-v1.5 512维语义检索",
    },
    "bm25_dense_rrf": {
        "label": "BM25 + Dense + RRF",
        "desc": "BM25 + Qdrant Dense + SQLite FTS5 三路 RRF(k=60)融合",
    },
    "full_pipeline": {
        "label": "五路融合 (当前方案)",
        "desc": "三路RRF + HyDE + MQE + heading_path加权",
    },
}
