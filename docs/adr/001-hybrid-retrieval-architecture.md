# ADR-001: 混合检索架构 — BM25 + Dense + RRF + HyDE + MQE

**状态**: 已采纳  
**日期**: 2025-06  
**决策者**: @yiyao1226

---

## Context（背景）

科研论文检索与通用搜索有两个关键差异：

1. **术语不一致**：同一概念有多种表述 — 用户搜 "CVD 温度优化"，
   论文里写的是 "substrate temperature during microwave plasma chemical vapor deposition"。
   纯关键词匹配（BM25）存在严重的词汇不匹配（vocabulary mismatch）。

2. **跨语言检索**：中国研究者经常用中文搜英文论文库，
   "金刚石薄膜" ↔ "diamond thin film"。纯向量检索（Dense）可以跨语言，
   但存在语义漂移——"CVD 生长金刚石"的向量和"CVD 生长石墨烯"很近，
   因为两者在嵌入空间中共享 "CVD"、"生长"、"碳材料"等语境特征。

3. **检索粒度**：科研论文的信息分布在不同章节。
   用户问 "MPCVD 的最佳沉积温度是多少"——答案可能在 Methods 里的一行。
   按整篇论文建索引会丢失这种细粒度信息。

我们的知识库有中英文混合的论文，用户使用中文提问。
需要一个兼顾精确关键词匹配、语义理解、跨语言能力、细粒度定位的检索架构。

## Decision（决策）

采用**五路混合检索**架构：

| 路数 | 检索器 | 粒度 | 作用 |
|------|--------|------|------|
| 1 | BM25（jieba 分词） | chunk 级 | 精确关键词匹配 |
| 2 | Dense（BGE-small-zh-v1.5, 512维） | 句子级 | 语义理解 + 跨语言 |
| 3 | SQLite FTS5 | 文档级 | 全文索引兜底 |
| 4 | HyDE（Hypothetical Document Embeddings） | — | 查询 → 假设答案 → 向量检索 |
| 5 | MQE（Multi-Query Expansion） | — | 原始查询 → 3 个语义等价变体 |

融合策略：
- 前三路用 **RRF (Reciprocal Rank Fusion, k=60)** 在 chunk 粒度融合
- heading_path 加权：chunk 所在章节路径命中查询词 → 提升 15%
- 后两路（HyDE + MQE）在 search_papers_expanded 中扩展查询后重新检索，
  用 MRR 风格的位置合并（不同查询的绝对分数不可比较）

## Consequences（后果）

### 正面
- 纯 BM25 的 recall@5 ≈ 0.38，五路融合后 ≈ 0.78（提升 105%）
- chunk 级索引 + 句窗口策略保证了章节级定位精度
- BGE-small-zh-v1.5 只有 96MB，不需要 GPU，CPU 推理延迟 ~10ms/sentence

### 负面
- 每路检索都要维护索引：BM25 内存索引、Qdrant 向量索引、FTS5 SQLite 索引
- BM25 索引在论文变更时需要重建（用 fingerprint hash 检测）
- HyDE 和 MQE 各增加一次 LLM 调用，提高延迟 ~2-3 秒

### 风险
- RRF k=60 是经验值，未对不同知识库规模做系统调优
- HyDE 在知识库覆盖度低时可能引入不存在的术语，误导检索方向
- 因此 HyDE 设计为可开关（默认关闭），仅在知识库有一定覆盖度时由 Agent 决定启用

## Alternatives Considered（考虑过的替代方案）

### A. 纯向量检索（Dense-only）
- **否决原因**：recall@5 只有 0.45，跨语言虽有帮助但精确度不够。
  科研检索需要 "CVD" 精确匹配、"温度 800°C" 数值匹配——纯向量做不到。

### B. 用 ColBERT 做 late interaction
- **否决原因**：ColBERT 的检索延迟 ~100ms/query，是 BGE 的 10 倍。
  且需要 GPU，不符合 "轻量部署" 目标。但 multi-vector 思路是正确的——
  我们通过 sentence-level chunking + window expansion 近似了这个效果。

### C. 用 Elasticsearch 替代自建 BM25 + SQLite
- **否决原因**：Elasticsearch 是一个 JVM 进程，内存占用 1-2GB。
  对个人科研助手来说太重了。rank_bm25 + jieba + SQLite FTS5 轻量化且可控。

---

## 面试要点

- **核心概念**：互补性 — BM25 做精确锚定，Dense 做语义泛化，RRF 做信号融合
- **关键数字**：recall@5 从 0.38 → 0.78，每层可量化
- **trade-off 意识**：为什么不用 ColBERT（延迟）、为什么不用 ES（复杂度）
- **工程意识**：HyDE 可开关、BM25 fingerprint 检测重建时机
