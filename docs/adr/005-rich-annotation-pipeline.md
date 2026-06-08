# ADR-005: 入库 LLM 厚标注策略

**状态**: 已采纳  
**日期**: 2025-06  
**决策者**: @yiyao1226

---

## Context（背景）

论文入库是 RAG 质量的第一道关卡。搜索引擎返回的论文元数据通常只有标题、作者、摘要、年份。
这些信息不足以支撑精准检索：

1. **摘要不完整**：摘要说的是"我们研究了 X"，但不会列出所有实验参数
2. **术语不一致**：论文用 "MWCVD"，用户搜 "微波等离子体化学气相沉积"
3. **方法细节隐式**："用 XRD 表征了薄膜质量"——但 "XRD" 是缩写，全文可能是 "X-ray diffraction (Bruker D8 Advance)"

如果直接把标题+摘要 embed → 存入 Qdrant → 检索时 semantic search，
结果是：大量论文的向量分布在同一片语义区域（因为摘要的表述模式高度相似），
区分度不够。

**解决思路**：入库前让 LLM 对论文做一次"厚标注"——提取结构化关键词、方法细节、
关键发现等元数据，混入嵌入文本，提高检索区分度。

## Decision（决策）

采用**LLM 一次标注 + 完整性验证 + 不足补全**的三段式入库管线：

### 标注 Schema

```json
{
  "keywords_material": ["金刚石薄膜", "diamond thin film"],  // 材料/化合物（中英双语）
  "keywords_method": ["MPCVD", "微波等离子体化学气相沉积"],  // 方法/工艺
  "keywords_phenomenon": ["表面粗糙度", "拉曼半峰宽", ...], // 现象/指标
  "key_findings": ["850°C 时拉曼半峰宽最小 (3.2 cm⁻¹)"],    // 定量发现
  "core_claim": "基片温度 850°C 为最优沉积温度",              // 一句话核心
  "contribution_type": "experiment",                         // 论文类型
  "methods": [{
    "method_name": "拉曼光谱",
    "method_category": "characterization",
    "key_parameters": {"激光波长": "532 nm", ...},
    "what_it_measures": "薄膜结晶质量",
    "equipment": "Horiba LabRAM HR Evolution"
  }],
  "remaining_gap": "未研究 800°C 以下的温度区间",             // 遗留缺口
  "baseline_methods": ["热丝 CVD"],
  "improvement_over_baseline": "拉曼半峰宽降低 40%"
}
```

### 嵌入文本构建

```python
def build_embedding_text(paper, annotation):
    return " ".join([
        paper["title"],
        paper["abstract"],
        # 混入标注关键词——等效于手工做了 query expansion
        " ".join(annotation.get("keywords_material", [])),
        " ".join(annotation.get("keywords_method", [])),
        " ".join(annotation.get("key_findings", [])),
        annotation.get("core_claim", ""),
    ])
```

效果：原来 embed("标题 + 摘要 300 字")，现在 embed("标题 + 摘要 + 20+ 个关键词 + 5 个发现 + 核心结论")。
向量从 "模糊的语义区域" 变成 "关键词密集的精确点"——BM25 和 Dense 都能更好地命中。

### 标注质量保障

```
generate_annotation(paper, full_text)
  │
  ├─ full_text? → 提取关键章节（~2500 字）
  │   else → 仅用标题 + 摘要
  │
  ├─ LLM 标注（temperature=0.1, 低随机性保证一致性）
  │
  ├─ validate_annotation()
  │   ├─ methods >= 2?
  │   ├─ key_findings >= 3?
  │   ├─ keywords_material >= 3?
  │   └─ core_claim 非空?
  │
  └─ 不足 & full_text 场景 → 二次补全（targeted prompt, temperature=0.05）
       └─ 仍然不足 → 标记 low_quality=True（不影响入库，但影响后续检索权重）
```

### Token 消耗分析

```
无全文（仅摘要，最常见场景）:
  LLM 输入:  标题 + 摘要 ~500 tokens
  LLM 输出:  标注 JSON ~300 tokens
  合计:      ~800 tokens/篇

有全文:
  LLM 输入:  提取的关键章节 ~2500 chars ≈ 1200 tokens
  LLM 输出:  标注 JSON ~500 tokens
  合计:      ~1700 tokens/篇

如果补全:
  追加 ~800 tokens
```

## Consequences（后果）

### 正面
- 检索区分度显著提升：embedding 包含了人工标注级别的关键词密度
- BM25 路直接受益：关键词被混入向量文本 → Dense 路也能命中
- 一次性标注永久复用：后续所有检索受益于首次入库的标注质量
- 标注完整性验证：不会出现 "methods=[]" 的空标注入库

### 负面
- 每篇论文入库 800-2600 tokens 的 LLM 成本
- 标注质量依赖 LLM 能力（DeepSeek-V4 在科研领域足够）
- 中英混用的关键词可能引入噪音（同一材料的中英文被当作两个关键词）

### 风险
- 标注 prompt 是硬编码的中文 → 英文论文标注会有翻译噪音
- `validate_annotation` 的阈值（methods >= 2 等）是经验值，未经系统调优
- LLM 幻觉：可能编造不存在的参数值（虽然 temperature=0.1 降低了概率）

## Alternatives Considered

### A. 不标注，直接用标题+摘要
- **否决原因**：检索质量太差。摘要的表述模式高度相似（"本文研究了...结果表明..."），
  向量全部聚集在一小片区域，区分度极低。

### B. 用小模型（如 BERT 微调）替代 LLM 做标注
- **否决原因**：标注任务需要理解论文的全局结构（方法 vs 结果 vs 结论），
  这超出了小模型的上下文窗口。且训练 BERT 标注器需要标注数据 → 鸡生蛋问题。

### C. 用 LLM 标注，但存为结构化字段而非混入嵌入
- **否决原因**：结构化字段只能用于 filter（精确匹配），不能用于语义搜索（近似匹配）。
  混入嵌入文本后，BM25 和 Dense 都能利用标注信息。

---

## 面试要点

- **核心概念**：入库质量的 ROI — 花 800 token 做标注，换后续每次检索都受益
- **关键设计**：标注 → 验证 → 补全 三段式保证质量
- **trade-off**：LLM 成本 vs 检索质量 — 对高频检索场景，一次标注的收益巨大
- **为什么是 LLM 而不是 BERT**：标注需要长上下文理解论文结构
